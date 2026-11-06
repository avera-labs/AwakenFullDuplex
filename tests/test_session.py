"""The session state machine on a tiny random model (CPU, no graphs)."""

import torch

from awaken_chat.frame import AUDIO, TEXT
from awaken_chat.inference.params import DecodeParams, SessionLimits
from awaken_chat.inference.session import Decoders, DuplexSession, State
from awaken_chat.models.configuration import AwakenAudioConfig, AwakenConfig
from awaken_chat.models.model import AwakenModel

VOCAB = 151680  # the real text vocab, so the special ids exist


def tiny_model() -> AwakenModel:
    torch.manual_seed(0)
    audio = AwakenAudioConfig(
        encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=32, d_model=16, output_dim=32,
        codebook_size=AUDIO.pad + 1, group_size=5,
        crq_transformer_config={"model_type": "qwen3", "hidden_size": 32, "intermediate_size": 48,
                                "num_hidden_layers": 1, "num_attention_heads": 2,
                                "num_key_value_heads": 1, "head_dim": 16, "vocab_size": 8},
    )
    text = {"model_type": "qwen3", "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16, "vocab_size": VOCAB}
    model = AwakenModel(AwakenConfig(audio_config=audio, text_config=text)).eval()
    for p in model.parameters():
        p.data.normal_(0, 0.02)
    model.tie_weights()
    return model


def make_session(model, params=DecodeParams()):
    dec = Decoders(model, SessionLimits(backbone_len=64, depth_len=40), use_graph=False, compile_on=False)
    prompt = torch.tensor([[1, 2, 3, TEXT.bos]])
    return DuplexSession(model, dec, prompt, params=params)


def test_idle_frames_feed_pad():
    model = tiny_model()
    sess = make_session(model)
    feats = torch.zeros(5, 32)
    sess.next_text = torch.tensor([TEXT.pad])
    out = sess.step(feats)
    assert out.audio_ids == [AUDIO.pad] * 5
    assert out.text_id == TEXT.pad


def test_content_opens_utterance_and_pad_releases():
    model = tiny_model()
    sess = make_session(model, DecodeParams(text_temperature=0.0, audio_temperature=0.0))
    feats = torch.zeros(5, 32)
    sess.next_text = torch.tensor([1000])  # force content on the next frame
    out = sess.step(feats)
    assert out.audio_ids == [AUDIO.bos] * 5 and sess.state is State.VOICING
    assert sess.next_audio is not None and sess.next_audio.shape == (1, 5)
    # force the release
    sess.next_text = torch.tensor([TEXT.pad])
    out = sess.step(feats)  # feeds the predicted codes, samples PAD -> release scheduled
    assert all(AUDIO.is_code(t) or t == AUDIO.eos for t in out.audio_ids)
    sess.next_text = torch.tensor([TEXT.pad])
    sess._release()
    out = sess.step(feats)
    assert out.audio_ids == [AUDIO.eos] * 5 and sess.state is State.IDLE
    sess.next_text = torch.tensor([TEXT.pad])  # stay listening (random weights may re-onset)
    out = sess.step(feats)
    assert out.audio_ids == [AUDIO.pad] * 5


def test_force_eos_ends_utterance():
    model = tiny_model()
    sess = make_session(model, DecodeParams(text_temperature=0.0, audio_temperature=0.0))
    feats = torch.zeros(5, 32)
    sess.next_text = torch.tensor([1000])
    sess.step(feats)
    sess.force_eos()
    out = sess.step(feats)
    assert out.audio_ids == [AUDIO.eos] * 5 and out.text_id == TEXT.pad and sess.state is State.IDLE


def test_history_lengths():
    model = tiny_model()
    sess = make_session(model)
    for _ in range(6):
        sess.step(torch.zeros(5, 32))
    assert len(sess.text_history) == len(sess.audio_history) == sess.frame == 6
