"""Codec tokens -> 24 kHz speech with CosyVoice3's flow matching model and HiFT vocoder.

The weights come from the upstream Hugging Face repo (``FunAudioLLM/Fun-CosyVoice3-0.5B-2512``);
only ``flow.pt``, ``hift.pt`` and the flow estimator's ONNX export are fetched. The codec's own
text-to-token LLM is never used: Awaken Chat produces the codec tokens itself.

Two estimator back-ends:

* TensorRT (default when ``tensorrt`` is importable and a GPU is present): the DiT estimator
  runs as an fp16 engine built once per machine from the ONNX export. This is the only
  back-end fast enough for live use (one 200 ms frame costs ~30-35 ms of flow + HiFT).
* Eager PyTorch (``prefer_trt=False`` or TensorRT unavailable): same math, launch-bound,
  several times slower. Fine for offline rendering.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

from awaken_chat.codec.cosyvoice.utils.common import TrtContextWrapper
from awaken_chat.codec.cosyvoice.utils.file_utils import convert_onnx_to_trt
from awaken_chat.frame import AUDIO

logger = logging.getLogger(__name__)

CODEC_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
CODEC_FILES = ["flow.pt", "hift.pt", "flow.decoder.estimator.fp32.onnx"]
VOICE_FILES = ["campplus.onnx", "speech_tokenizer_v3.onnx"]
CONFIG_YAML = Path(__file__).with_name("cosyvoice") / "cosyvoice3.yaml"
TOKEN_HOP_LEN = 25  # flow chunk size in tokens (must match training)
SPK_EMBED_DIM = 192
MEL_DIM = 80


@dataclass
class VoicePrompt:
    """Flow conditioning that pins the timbre: x-vector, prompt tokens and their mel.

    ``prompt_feat`` has exactly ``2 * prompt_token.shape[1]`` frames. ``None`` as a voice means
    an unconditioned decode (zero x-vector, empty prompt).
    """

    prompt_token: torch.Tensor  # (1, N) int32
    prompt_feat: torch.Tensor  # (1, 2N, 80) float32
    embedding: torch.Tensor  # (1, 192) float32

    def to(self, device) -> VoicePrompt:
        return VoicePrompt(self.prompt_token.to(device).to(torch.int32),
                           self.prompt_feat.to(device).float(), self.embedding.to(device).float())

    @staticmethod
    def empty(device) -> VoicePrompt:
        return VoicePrompt(torch.zeros(1, 0, dtype=torch.int32, device=device),
                           torch.zeros(1, 0, MEL_DIM, device=device),
                           torch.zeros(1, SPK_EMBED_DIM, device=device))

    def save(self, path: str | Path) -> None:
        torch.save({"prompt_token": self.prompt_token.cpu(), "prompt_feat": self.prompt_feat.cpu(),
                    "embedding": self.embedding.cpu()}, path)

    @classmethod
    def load(cls, path: str | Path) -> VoicePrompt:
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(d["prompt_token"], d["prompt_feat"], d["embedding"])


def codec_dir(repo_or_dir: str = CODEC_REPO, with_voice_tools: bool = False) -> str:
    """A local codec directory as-is, otherwise the needed files of the Hub repo."""
    if os.path.isdir(repo_or_dir):
        return repo_or_dir
    from huggingface_hub import snapshot_download

    files = CODEC_FILES + (VOICE_FILES if with_voice_tools else [])
    return snapshot_download(repo_or_dir, allow_patterns=files)


def _trt_profile() -> dict:
    return {"min_shape": [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)],
            "opt_shape": [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)],
            "max_shape": [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)],
            "input_names": ["x", "mask", "mu", "cond"]}


def _load_trt_estimator(flow, model_dir: str, fp16: bool, device: torch.device,
                        plan: str | None = None) -> None:
    """Replace the eager DiT estimator with a TensorRT engine (built on first use).

    Plans are specific to a GPU model and TensorRT version, so the default file name carries
    the GPU name and the plan is built next to the weights the first time."""
    import tensorrt as trt

    precision = "fp16" if fp16 else "fp32"
    if plan is None:
        gpu = torch.cuda.get_device_name(device).replace(" ", "_")
        plan = os.path.join(model_dir, f"flow.decoder.estimator.{precision}.{gpu}.plan")
    onnx = os.path.join(model_dir, "flow.decoder.estimator.fp32.onnx")
    if not os.path.exists(plan) or os.path.getsize(plan) == 0:
        logger.info("building the TensorRT %s estimator plan (one-time, minutes): %s", precision, plan)
        convert_onnx_to_trt(plan, _trt_profile(), onnx, fp16)
    del flow.decoder.estimator
    with open(plan, "rb") as f:
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to load the TensorRT plan {plan}")
    flow.decoder.estimator = TrtContextWrapper(engine, trt_concurrent=1, device=device)


def _optimize_hift(hift) -> None:
    """Strip weight-norm, pin dtypes/formats and compile HiFT's decode for steady per-call cost."""
    from torch.nn.utils import remove_weight_norm
    from torch.nn.utils.parametrize import remove_parametrizations

    for module in hift.modules():
        if hasattr(module, "parametrizations") and hasattr(module.parametrizations, "weight"):
            remove_parametrizations(module, "weight")
        else:
            try:
                remove_weight_norm(module)
            except ValueError:
                pass
    hift.f0_predictor.to(torch.float64)
    hift.to(memory_format=torch.channels_last)
    hift.stft_window = hift.stft_window.to(next(hift.parameters()).device)
    # STFT/ISTFT are not compilable; the causal convs' data-dependent padding causes graph
    # breaks and a handful of recompiles that Dynamo handles on its own (this partitioning is
    # what the equivalence gate is pinned to). Those are expected, so their warnings are muted.
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
    logging.getLogger("torch._inductor").setLevel(logging.ERROR)
    hift._stft = torch._dynamo.disable(hift._stft)
    hift._istft = torch._dynamo.disable(hift._istft)
    hift.decode = torch.compile(hift.decode, mode="default")


class AudioDetokenizer:
    """CosyVoice3 flow + HiFT, loaded once per process."""

    def __init__(self, flow, hift, device: torch.device, fp16: bool, model_dir: str) -> None:
        self.flow, self.hift, self.device, self.fp16, self.model_dir = flow, hift, device, fp16, model_dir

    @classmethod
    def load(cls, repo_or_dir: str = CODEC_REPO, device: str | torch.device = "cuda",
             prefer_trt: bool = True, fp16: bool = True, optimize: bool = True,
             warmup: bool = True, trt_plan: str | None = None) -> AudioDetokenizer:
        """``trt_plan``: use this engine file instead of the default per-GPU one."""
        device = torch.device(device)
        model_dir = codec_dir(repo_or_dir)
        with open(CONFIG_YAML) as f:
            cfg = load_hyperpyyaml(f)
        flow, hift = cfg["flow"], cfg["hift"]
        flow.load_state_dict(torch.load(os.path.join(model_dir, "flow.pt"), map_location="cpu",
                                        weights_only=True), strict=True)
        hift_sd = torch.load(os.path.join(model_dir, "hift.pt"), map_location="cpu", weights_only=True)
        hift.load_state_dict({k.replace("generator.", ""): v for k, v in hift_sd.items()}, strict=True)
        flow.to(device).eval()
        hift.to(device).eval()
        flow.decoder.estimator.static_chunk_size = 2 * TOKEN_HOP_LEN

        use_trt = prefer_trt and device.type == "cuda"
        if use_trt:
            try:
                _load_trt_estimator(flow, model_dir, fp16, device, trt_plan)
                logger.info("flow estimator: TensorRT %s", "fp16" if fp16 else "fp32")
            except Exception as e:  # noqa: BLE001 - fall back to the eager estimator
                logger.warning("TensorRT estimator unavailable (%s); using the eager estimator", e)
                use_trt = False
        if not use_trt:
            fp16 = False
        if optimize and device.type == "cuda":
            _optimize_hift(hift)
        else:
            hift.stft_window = hift.stft_window.to(device)
        obj = cls(flow, hift, device, fp16, model_dir)
        if warmup:
            obj.warmup()
        return obj

    # Decoding -------------------------------------------------------------------------
    def _flow(self, tokens: torch.Tensor, voice: VoicePrompt, finalize: bool) -> torch.Tensor:
        """Tokens (1, N) + voice -> mel (1, 80, 2N) of the tokens (prompt mel excluded)."""
        dev = self.device
        with torch.no_grad(), torch.autocast("cuda", enabled=self.fp16):
            mel, _ = self.flow.inference(
                token=tokens, token_len=torch.tensor([tokens.shape[1]], dtype=torch.int32, device=dev),
                prompt_token=voice.prompt_token,
                prompt_token_len=torch.tensor([voice.prompt_token.shape[1]], dtype=torch.int32, device=dev),
                prompt_feat=voice.prompt_feat,
                prompt_feat_len=torch.tensor([voice.prompt_feat.shape[1]], dtype=torch.int32, device=dev),
                embedding=voice.embedding, streaming=True, finalize=finalize,
            )
        return mel

    def _vocode(self, mel: torch.Tensor, finalize: bool) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=self.fp16):
            speech, _ = self.hift.inference(speech_feat=mel, finalize=finalize)
        return speech

    def decode_tokens(self, token_ids: torch.Tensor, voice: VoicePrompt | None = None) -> torch.Tensor:
        """One-shot decode of a 1-D token tensor -> 1-D float32 waveform at 24 kHz."""
        token_ids = token_ids.where(token_ids < AUDIO.codebook_size, torch.ones_like(token_ids))
        tokens = token_ids[None].to(self.device, torch.int32)
        v = voice.to(self.device) if voice is not None else VoicePrompt.empty(self.device)
        mel = self._flow(tokens, v, finalize=True)
        return self._vocode(mel, finalize=True).squeeze(0).float().cpu()

    def stream_session(self, voice: VoicePrompt | None = None, **kwargs):
        from awaken_chat.codec.streaming import StreamingDetokSession

        return StreamingDetokSession(self, voice, **kwargs)

    def warmup(self, n: int = 30) -> None:
        """Run flow + HiFT a few times so compilation and allocator caches are paid up front."""
        logger.info("warming up the detokenizer")
        dummy = torch.randint(0, AUDIO.codebook_size, (5,), device=self.device)
        for _ in range(n):
            self.decode_tokens(dummy)
        torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
