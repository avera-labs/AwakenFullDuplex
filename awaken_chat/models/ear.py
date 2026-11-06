"""The ear: a causal Whisper-style encoder that turns user audio into per-frame features.

Two ways to run it, with identical semantics:

* :class:`StreamingEar` — live use. Feed 24 kHz PCM as it arrives; it returns one
  ``(GROUP_SIZE, output_dim)`` feature tensor per completed 200 ms frame. Mel columns are
  finalized with a small settle horizon, and the tower runs incrementally with per-layer
  key/value caches trimmed to the 2 s attention window.
* :meth:`CausalAudioEncoder.encode_track` — offline use. Encode a whole 16 kHz mel track in
  one pass with causal windowed attention. The tower is causal, so the result is prefix-stable
  and matches the streaming path.

Architecture (weights inherited from Fun-Audio-Chat, then trained end to end with a causal
2 s window and a positional embedding that repeats every 100 tower tokens):

    mel (128, T) -> conv1 (k3) -> GELU -> conv2 (k3, stride 2) -> GELU -> + sinusoid[pos % 100]
        -> 32 x [LN -> causal windowed self-attention -> LN -> MLP] -> avg-pool 2 -> LN -> proj
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from transformers import WhisperFeatureExtractor
from transformers.activations import ACT2FN
from transformers.audio_utils import spectrogram, window_function

from awaken_chat.frame import (
    EAR_ATTN_WINDOW,
    EAR_POS_PERIOD,
    EAR_TOKENS_PER_FRAME,
    MEL_BINS,
    MEL_PER_FRAME,
    SR_IN,
    SR_MEL,
)
from awaken_chat.models.configuration import AwakenAudioConfig

try:  # optional: fused causal windowed attention for the offline whole-track pass
    from flash_attn import flash_attn_varlen_func
except ImportError:  # pragma: no cover - exercised on machines without flash-attn
    flash_attn_varlen_func = None


class EarAttention(nn.Module):
    """Multi-head self-attention of one encoder layer (projection weights only).

    The attention pattern itself (causal, windowed) is applied by the caller:
    :class:`StreamingEar` keeps per-layer caches, :meth:`CausalAudioEncoder.encode_track`
    uses a fused varlen kernel.
    """

    def __init__(self, config: AwakenAudioConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("d_model must be divisible by encoder_attention_heads")
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x`` (L, D) -> q, k, v each (L, heads, head_dim)."""
        length = x.shape[0]
        q = self.q_proj(x).reshape(length, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(length, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(length, self.num_heads, self.head_dim)
        return q, k, v


class EarLayer(nn.Module):
    """Pre-norm transformer layer: ``x + attn(ln1(x))`` then ``x + mlp(ln2(x))``."""

    def __init__(self, config: AwakenAudioConfig) -> None:
        super().__init__()
        self.self_attn = EarAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(self.final_layer_norm(x))))


class SinusoidPositions(nn.Module):
    """Whisper's fixed sinusoidal table, ``(length, channels)``, not saved with the weights."""

    def __init__(self, length: int, channels: int, max_timescale: float = 10_000) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("channels must be even")
        log_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_increment * torch.arange(channels // 2).float())
        scaled = torch.arange(length)[:, None] * inv_timescales[None, :]
        table = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)
        self.register_buffer("positional_embedding", table, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.positional_embedding[positions]


class CausalAudioEncoder(nn.Module):
    """The ear's weights and its offline whole-track forward."""

    def __init__(self, config: AwakenAudioConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.conv1 = nn.Conv1d(config.num_mel_bins, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList(EarLayer(config) for _ in range(config.encoder_layers))
        self.ln_post = nn.LayerNorm(d)
        self.proj = nn.Linear(d, config.output_dim)
        self.positional_embedding = SinusoidPositions(config.max_source_positions, d)
        # Present in the checkpoint (inherited), unused by the duplex stream.
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)

    @property
    def device(self) -> torch.device:
        return self.proj.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.proj.weight.dtype

    def front_end(self, mel: torch.Tensor) -> torch.Tensor:
        """``mel`` (B, 128, T) -> tower tokens (B, T // 2, D) with repeating positions."""
        x = F.gelu(self.conv1(mel.to(self.device, self.dtype)))
        x = F.gelu(self.conv2(x)).transpose(1, 2)
        positions = torch.arange(x.shape[1], device=self.device) % EAR_POS_PERIOD
        return x + self.positional_embedding(positions).to(self.dtype)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """Tower output (L, D) -> features (L // 2, output_dim): pool, norm, project."""
        pooled = F.avg_pool1d(h.T[None].float(), 2, 2)[0].T.to(self.dtype)
        return self.proj(self.ln_post(pooled))

    @torch.no_grad()
    def encode_track(self, mel: torch.Tensor) -> torch.Tensor:
        """Whole-track causal encode of one 16 kHz mel ``(128, T)``, ``T`` a multiple of 20.

        Returns ``(T // 20, GROUP_SIZE, output_dim)``: one feature group per frame. Uses the
        fused varlen kernel when flash-attn is installed, otherwise the exact streaming tower.
        """
        if mel.shape[1] % MEL_PER_FRAME:
            raise ValueError(f"mel length {mel.shape[1]} is not a multiple of {MEL_PER_FRAME}")
        n_frames = mel.shape[1] // MEL_PER_FRAME
        if flash_attn_varlen_func is None or self.device.type != "cuda":
            tower = IncrementalTower(self)
            feats = [tower.step(mel, f) for f in range(n_frames)]
            return torch.stack(feats)
        h = self.front_end(mel[None])[0]
        length = h.shape[0]
        cu = torch.tensor([0, length], dtype=torch.int32, device=self.device)
        for layer in self.layers:
            q, k, v = layer.self_attn.qkv(layer.self_attn_layer_norm(h))
            a = flash_attn_varlen_func(
                q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=length,
                max_seqlen_k=length, softmax_scale=layer.self_attn.scaling, causal=True,
                window_size=(EAR_ATTN_WINDOW, 0),
            )
            h = h + layer.self_attn.out_proj(a.reshape(length, -1))
            h = h + layer.mlp(h)
        return self.head(h).reshape(n_frames, -1, self.config.output_dim)


# --------------------------------------------------------------------------------------
# Streaming path
# --------------------------------------------------------------------------------------

CONV_WAIT_COLS = 2  # the two kernel-3 convs need two future mel columns
SETTLE_COLS = 3  # a mel column is final once 3 later columns exist (running-max floor)
SEED_LOGMAX = 0.8  # floor seed so the first columns are not normalized against silence
RESAMPLE_MARGIN_IN = 2400  # left context for the chunked 24k -> 16k resample
RESAMPLE_HOLD_IN = 720  # right hold-back (sinc lookahead), 30 ms
MAX_SESSION_S = 330  # preallocated buffer horizon


class StreamingMel:
    """Whisper log-mel computed column by column with a running-max floor.

    Whisper normalizes a clip against its global maximum. Live audio has no global maximum,
    so the floor is the running maximum over all columns seen so far (seeded at
    ``SEED_LOGMAX``), and a column is finalized only once ``SETTLE_COLS`` later columns exist.
    """

    def __init__(self, feature_extractor: WhisperFeatureExtractor) -> None:
        self.fe = feature_extractor
        cap16 = MAX_SESSION_S * SR_MEL
        cap_cols = MAX_SESSION_S * 100 + 16
        self.buf16 = np.zeros(cap16, dtype=np.float32)
        self.n16 = 0
        self.raw = np.zeros((MEL_BINS, cap_cols), dtype=np.float32)
        self.n_raw = 0
        self.colmax = np.zeros(cap_cols, dtype=np.float32)
        self.melbuf = np.zeros((MEL_BINS, cap_cols), dtype=np.float32)
        self.n_mel = 0
        self.window = window_function(feature_extractor.n_fft, "hann")

    @property
    def mel(self) -> torch.Tensor:
        """All finalized columns, ``(128, n_mel)``."""
        return torch.from_numpy(self.melbuf[:, : self.n_mel])

    def feed(self, pcm16k: np.ndarray) -> torch.Tensor:
        """Append 16 kHz samples; returns the finalized mel so far."""
        n = len(pcm16k)
        self.buf16[self.n16 : self.n16 + n] = pcm16k
        self.n16 += n
        fe = self.fe
        t_avail = (self.n16 - fe.n_fft // 2) // fe.hop_length
        if t_avail > self.n_raw:
            done = self.n_raw
            lo_col = max(0, done - 4)
            lo = lo_col * fe.hop_length
            raw = spectrogram(
                self.buf16[lo : self.n16], self.window, frame_length=fe.n_fft,
                hop_length=fe.hop_length, power=2.0, mel_filters=fe.mel_filters,
                log_mel="log10",
            )
            raw = np.asarray(raw)
            new = raw[:, done - lo_col : t_avail - lo_col]
            self.raw[:, done:t_avail] = new
            prev = self.colmax[done - 1] if done else -np.inf
            self.colmax[done:t_avail] = np.maximum.accumulate(np.maximum(new.max(axis=0), prev))
            self.n_raw = t_avail
        can_final = self.n_raw - SETTLE_COLS
        if can_final > self.n_mel:
            a, b = self.n_mel, can_final
            floor = np.maximum(self.colmax[a + SETTLE_COLS : b + SETTLE_COLS], SEED_LOGMAX) - 8.0
            cols = np.maximum(self.raw[:, a:b], floor[None, :])
            self.melbuf[:, a:b] = (cols + 4.0) / 4.0
            self.n_mel = b
        return self.mel


class IncrementalTower:
    """Frame-by-frame forward of :class:`CausalAudioEncoder` with per-layer KV caches.

    The conv front-end is recomputed over the full mel prefix (cheap and exact); attention
    appends ``EAR_TOKENS_PER_FRAME`` tokens per frame and keeps the last ``EAR_ATTN_WINDOW``
    keys per layer. Attention runs in fp32.
    """

    def __init__(self, tower: CausalAudioEncoder) -> None:
        self.tower = tower
        self.device, self.dtype = tower.device, tower.dtype
        attn = tower.layers[0].self_attn
        self.heads, self.head_dim = attn.num_heads, attn.head_dim
        empty = torch.zeros(0, self.heads, self.head_dim, device=self.device, dtype=self.dtype)
        self.caches = [[empty, empty] for _ in tower.layers]
        self.frames_done = 0

    @torch.no_grad()
    def _attend(self, layer_index: int, x: torch.Tensor, q_start: int) -> torch.Tensor:
        attn = self.tower.layers[layer_index].self_attn
        length = x.shape[0]
        q, k, v = attn.qkv(x)
        cache_k, cache_v = self.caches[layer_index]
        k_all = torch.cat([cache_k, k], 0)
        v_all = torch.cat([cache_v, v], 0)
        kv_offset = q_start - cache_k.shape[0]
        qi = torch.arange(q_start, q_start + length, device=self.device)[:, None]
        kj = torch.arange(kv_offset, kv_offset + k_all.shape[0], device=self.device)[None, :]
        mask = (kj <= qi) & (kj >= qi - EAR_ATTN_WINDOW)
        scores = torch.einsum("lhd,shd->hls", q.float(), k_all.float()) * attn.scaling
        scores = scores.masked_fill(~mask[None], float("-inf"))
        out = torch.einsum("hls,shd->lhd", scores.softmax(-1), v_all.float())
        self.caches[layer_index] = [k_all[-EAR_ATTN_WINDOW:], v_all[-EAR_ATTN_WINDOW:]]
        return attn.out_proj(out.reshape(length, -1).to(self.dtype))

    @torch.no_grad()
    def step(self, mel: torch.Tensor, frame: int) -> torch.Tensor:
        """Features ``(GROUP_SIZE, output_dim)`` of ``frame``; frames must be stepped in order.

        ``mel`` holds at least ``20 * (frame + 1) + 2`` finalized columns (the ``+ 2`` is the
        conv lookahead; at the very end of a track fewer columns are accepted).
        """
        if frame != self.frames_done:
            raise ValueError(f"expected frame {self.frames_done}, got {frame}")
        tower = self.tower
        end = min(mel.shape[1], (frame + 1) * MEL_PER_FRAME + CONV_WAIT_COLS)
        if end < (frame + 1) * MEL_PER_FRAME:
            raise ValueError("not enough mel columns for this frame")
        x = tower.front_end(mel[None, :, :end])[0]
        lo, hi = frame * EAR_TOKENS_PER_FRAME, (frame + 1) * EAR_TOKENS_PER_FRAME
        h = x[lo:hi]
        for li, layer in enumerate(tower.layers):
            h = h + self._attend(li, layer.self_attn_layer_norm(h), lo)
            h = h + layer.mlp(h)
        self.frames_done += 1
        return tower.head(h)


class StreamingEar:
    """24 kHz microphone PCM in, one feature group per completed frame out.

    Frame ``f`` becomes computable once the input holds ``(f + 1) * 4800`` samples plus the
    hold-back (resample sinc lookahead + mel settle + conv wait, about 60 ms).
    """

    def __init__(self, tower: CausalAudioEncoder, feature_extractor: WhisperFeatureExtractor):
        self.track = np.zeros(MAX_SESSION_S * SR_IN, dtype=np.float32)
        self.n_track = 0
        self.mel = StreamingMel(feature_extractor)
        self.tower = IncrementalTower(tower)
        self.resampled = 0  # 24 kHz samples already resampled and fed to the mel
        self.next_frame = 0

    def feed(self, pcm24: np.ndarray) -> list[torch.Tensor]:
        """Append samples; returns the features of every frame that became computable."""
        n = len(pcm24)
        self.track[self.n_track : self.n_track + n] = pcm24
        self.n_track += n
        # Chunked 24k -> 16k resample with left margin and right hold-back so every emitted
        # 16k sample has full sinc context; boundaries stay on multiples of 3 (2:3 ratio).
        target = self.n_track - RESAMPLE_HOLD_IN
        target -= target % 3
        if target > self.resampled:
            lo = max(0, self.resampled - RESAMPLE_MARGIN_IN)
            lo -= lo % 3
            seg16 = torchaudio.functional.resample(
                torch.from_numpy(self.track[lo:target]), SR_IN, SR_MEL
            ).numpy()
            self.mel.feed(seg16[(self.resampled - lo) * 2 // 3 :])
            self.resampled = target
        mel = self.mel.mel
        out = []
        while mel.shape[1] >= (self.next_frame + 1) * MEL_PER_FRAME + CONV_WAIT_COLS:
            out.append(self.tower.step(mel, self.next_frame))
            self.next_frame += 1
        return out


def mel_of_track(track16: np.ndarray, feature_extractor: WhisperFeatureExtractor,
                 n_frames: int) -> torch.Tensor:
    """Offline mel ``(128, 20 * n_frames)`` of a 16 kHz track (padded or cut to the span).

    Uses the stock Whisper extractor (global-max normalization), the recipe the offline
    benchmarks run; the streaming recipe differs only in its running floor.
    """
    target = n_frames * (SR_MEL // 5)
    track16 = np.asarray(track16, dtype=np.float32)
    if track16.shape[0] < target:
        track16 = np.concatenate([track16, np.zeros(target - track16.shape[0], np.float32)])
    track16 = track16[:target]
    fe = feature_extractor
    if target > fe.n_samples:  # lift the extractor's 30 s/300 s truncation cap
        fe.chunk_length = math.ceil(target / fe.sampling_rate)
        fe.n_samples = fe.chunk_length * fe.sampling_rate
        fe.nb_max_frames = fe.n_samples // fe.hop_length
    feats = fe(track16, sampling_rate=SR_MEL, padding="longest")["input_features"][0]
    return torch.from_numpy(np.asarray(feats))[:, : n_frames * MEL_PER_FRAME]
