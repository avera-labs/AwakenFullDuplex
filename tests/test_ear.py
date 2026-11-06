"""The streaming ear equals a direct causal-windowed forward (tiny random tower, CPU)."""

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from awaken_chat.frame import EAR_ATTN_WINDOW, EAR_POS_PERIOD, MEL_PER_FRAME
from awaken_chat.models.configuration import AwakenAudioConfig
from awaken_chat.models.ear import CausalAudioEncoder, IncrementalTower, StreamingEar, mel_of_track


def tiny_ear(seed: int = 0) -> CausalAudioEncoder:
    torch.manual_seed(seed)
    cfg = AwakenAudioConfig(encoder_layers=2, encoder_attention_heads=4, encoder_ffn_dim=96,
                            d_model=32, output_dim=24, max_source_positions=1500)
    return CausalAudioEncoder(cfg).eval()


@torch.no_grad()
def reference_forward(ear: CausalAudioEncoder, mel: torch.Tensor) -> torch.Tensor:
    """Whole-sequence forward with an explicit causal window mask."""
    h = ear.front_end(mel[None])[0]
    n = h.shape[0]
    i = torch.arange(n)[:, None]
    j = torch.arange(n)[None, :]
    allowed = (j <= i) & (j >= i - EAR_ATTN_WINDOW)
    for layer in ear.layers:
        q, k, v = layer.self_attn.qkv(layer.self_attn_layer_norm(h))
        s = torch.einsum("lhd,shd->hls", q, k) * layer.self_attn.scaling
        s = s.masked_fill(~allowed[None], float("-inf"))
        a = torch.einsum("hls,shd->lhd", s.softmax(-1), v).reshape(n, -1)
        h = h + layer.self_attn.out_proj(a)
        h = h + layer.mlp(h)
    return ear.head(h)


def test_incremental_tower_matches_reference():
    ear = tiny_ear()
    n_frames = 30  # 6 s: beyond one position period and one attention window
    mel = torch.randn(128, n_frames * MEL_PER_FRAME + 2)
    # The reference sees the same two lookahead columns the incremental conv front-end uses;
    # the extra tower token they produce is dropped.
    ref = reference_forward(ear, mel)[: n_frames * MEL_PER_FRAME // 4]
    tower = IncrementalTower(ear)
    out = torch.cat([tower.step(mel, f) for f in range(n_frames)])
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    assert n_frames * MEL_PER_FRAME // 2 > max(EAR_POS_PERIOD, EAR_ATTN_WINDOW)


def test_encode_track_matches_incremental():
    ear = tiny_ear(1)
    mel = torch.randn(128, 20 * MEL_PER_FRAME)
    whole = ear.encode_track(mel)
    tower = IncrementalTower(ear)
    inc = torch.stack([tower.step(mel, f) for f in range(20)])
    torch.testing.assert_close(whole, inc, atol=1e-4, rtol=1e-4)


def test_streaming_ear_emits_one_group_per_frame():
    ear = tiny_ear(2)
    fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000)
    sear = StreamingEar(ear, fe)
    rng = np.random.default_rng(0)
    total = 0
    for _ in range(25):  # 25 chunks of 100 ms = 2.5 s
        feats = sear.feed(rng.standard_normal(2400).astype(np.float32) * 0.1)
        total += len(feats)
        for f in feats:
            assert f.shape == (5, 24)
    # 2.5 s = 12.5 frames; the hold-back costs a bit of lookahead
    assert 10 <= total <= 12


def test_mel_of_track_shape():
    fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000)
    mel = mel_of_track(np.zeros(16000, np.float32), fe, n_frames=10)
    assert mel.shape == (128, 200)
