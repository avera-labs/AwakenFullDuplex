"""Voice prompts: extract the flow's timbre conditioning from a reference clip.

A :class:`~awaken_chat.codec.detokenizer.VoicePrompt` is three tensors the flow was trained to
condition on (zero-shot voice cloning): a CAM++ x-vector, the clip's codec tokens from the
CosyVoice3 speech tokenizer, and the clip's 24 kHz mel (2 frames per token).

Extraction needs the two upstream ONNX models (``campplus.onnx``, ``speech_tokenizer_v3.onnx``)
and ``onnxruntime`` (``pip install awaken-chat[voice]``). Prompts are small; save them with
:meth:`VoicePrompt.save` and ship the ``.pt`` so the runtime needs none of this.

Only use clips of speakers whose voice you have the right to use.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import WhisperFeatureExtractor
from transformers.audio_utils import mel_filter_bank

from awaken_chat.codec.detokenizer import CODEC_REPO, VoicePrompt, codec_dir
from awaken_chat.frame import SR_IN, SR_MEL

TOKENS_PER_SECOND = 25
FLOW_MEL = {"n_fft": 1920, "hop": 480, "win": 1920, "n_mels": 80}  # the flow's 24 kHz mel


def load_clip(path: str | Path, sample_rate: int, max_s: float) -> torch.Tensor:
    """Mono ``(1, T)`` float tensor at ``sample_rate``, cut to ``max_s`` with a 50 ms fade-out.

    The fade matters: a hard cut mid-speech leaves the prompt mel ending on a non-zero
    waveform, and the causal vocoder then starts every utterance with a click.
    """
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = torch.from_numpy(wav.mean(axis=1))
    if sr != sample_rate:
        x = torchaudio.functional.resample(x, sr, sample_rate)
    x = x[: int(max_s * sample_rate)]
    n = int(0.05 * sample_rate)
    if x.shape[0] > n:
        x = x.clone()
        x[-n:] *= torch.linspace(1.0, 0.0, n)
    return x[None]


def flow_mel(wav24: torch.Tensor) -> torch.Tensor:
    """The flow's mel of a 24 kHz clip ``(1, T)`` -> ``(frames, 80)`` (HiFi-GAN recipe)."""
    p = FLOW_MEL
    filters = torch.from_numpy(mel_filter_bank(
        num_frequency_bins=p["n_fft"] // 2 + 1, num_mel_filters=p["n_mels"], min_frequency=0.0,
        max_frequency=SR_IN / 2, sampling_rate=SR_IN, norm="slaney", mel_scale="slaney",
    )).float()
    pad = (p["n_fft"] - p["hop"]) // 2
    y = torch.nn.functional.pad(wav24[None], (pad, pad), mode="reflect")[0]
    spec = torch.stft(y, p["n_fft"], hop_length=p["hop"], win_length=p["win"],
                      window=torch.hann_window(p["win"]), center=False, return_complex=True)
    mag = torch.sqrt(spec.abs().pow(2) + 1e-9)
    mel = torch.log(torch.clamp(filters.T @ mag[0], min=1e-5))  # (80, frames)
    return mel.T


class VoiceExtractor:
    def __init__(self, repo_or_dir: str = CODEC_REPO) -> None:
        import onnxruntime

        d = codec_dir(repo_or_dir, with_voice_tools=True)
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        self.campplus = onnxruntime.InferenceSession(os.path.join(d, "campplus.onnx"), opts,
                                                     providers=["CPUExecutionProvider"])
        self.tokenizer = onnxruntime.InferenceSession(os.path.join(d, "speech_tokenizer_v3.onnx"), opts,
                                                      providers=["CPUExecutionProvider"])
        self.whisper_fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=SR_MEL)

    def extract(self, clip: str | Path, max_s: float = 6.0) -> VoicePrompt:
        import torchaudio.compliance.kaldi as kaldi

        wav16 = load_clip(clip, SR_MEL, max_s)
        wav24 = load_clip(clip, SR_IN, max_s)
        # speech tokens from the 128-bin Whisper log-mel (same recipe as whisper.log_mel_spectrogram)
        mel128 = self.whisper_fe(wav16[0].numpy(), sampling_rate=SR_MEL, padding="longest",
                                 return_tensors="np")["input_features"]
        inputs = self.tokenizer.get_inputs()
        tokens = self.tokenizer.run(None, {inputs[0].name: mel128.astype(np.float32),
                                           inputs[1].name: np.array([mel128.shape[2]], dtype=np.int32)})[0]
        prompt_token = torch.tensor(tokens.flatten()[None], dtype=torch.int32)
        # x-vector from an 80-bin kaldi fbank
        fbank = kaldi.fbank(wav16, num_mel_bins=80, dither=0, sample_frequency=SR_MEL)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        emb = self.campplus.run(None, {self.campplus.get_inputs()[0].name: fbank[None].numpy()})[0]
        # prompt mel aligned to 2 frames per token
        feat = flow_mel(wav24)
        n = min(feat.shape[0] // 2, prompt_token.shape[1])
        return VoicePrompt(prompt_token[:, :n], feat[: 2 * n][None].contiguous(),
                           torch.tensor(emb.reshape(1, -1), dtype=torch.float32))


def from_spk2info(entry: dict) -> VoicePrompt:
    """A CosyVoice ``spk2info.pt`` preset (embedding, speech_token, speech_feat) -> VoicePrompt.

    Stock presets ship with mel and tokens slightly misaligned; the prompt keeps the aligned
    prefix (2 mel frames per token).
    """
    tok = entry["speech_token"].to(torch.int32)
    feat = entry["speech_feat"].float()
    n = min(feat.shape[1] // 2, tok.shape[1])
    return VoicePrompt(tok[:, :n], feat[:, : 2 * n].contiguous(), entry["embedding"].float().reshape(1, -1))


def trim_prompt(voice: VoicePrompt, max_s: float) -> VoicePrompt:
    """Keep the last ``max_s`` seconds of a prompt (the prompt sits in every flow call)."""
    n = voice.prompt_token.shape[1]
    keep = min(n, int(max_s * TOKENS_PER_SECOND))
    return VoicePrompt(voice.prompt_token[:, n - keep:], voice.prompt_feat[:, 2 * (n - keep):],
                       voice.embedding)
