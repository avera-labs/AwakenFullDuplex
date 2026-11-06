"""StaticDecoder == transformers' Qwen3 forward (tiny random model, CPU, no graphs)."""

import torch
from transformers import DynamicCache, Qwen3Config, Qwen3Model

from awaken_chat.inference.static_decoder import StaticDecoder


def tiny_qwen3(seed: int = 0) -> Qwen3Model:
    torch.manual_seed(seed)
    cfg = Qwen3Config(hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      vocab_size=50, max_position_embeddings=256)
    return Qwen3Model(cfg).eval()


@torch.no_grad()
def test_matches_hf_with_prefix_and_steps():
    m = tiny_qwen3()
    dec = StaticDecoder(m, max_len=32, use_graph=False, compile_on=False)
    prefix = torch.randn(1, 5, 64)
    out = m(inputs_embeds=prefix, use_cache=True, return_dict=True)
    cache = out.past_key_values
    dec.load_prefix(cache, 5)
    ref_cache = DynamicCache()
    m(inputs_embeds=prefix, past_key_values=ref_cache, use_cache=True)
    for step in range(6):
        x = torch.randn(1, 64)
        pos = 5 + step
        ref = m(inputs_embeds=x[None], past_key_values=ref_cache, use_cache=True,
                position_ids=torch.tensor([[pos]]), return_dict=True)
        h, _ = dec.step(x, pos)
        torch.testing.assert_close(h, ref.last_hidden_state[:, -1], atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_block_reset_restarts_attention():
    """After reset() the decoder attends only to the new block, like a fresh cache."""
    m = tiny_qwen3(1)
    dec = StaticDecoder(m, max_len=16, use_graph=False, compile_on=False)
    for i in range(4):
        dec.step(torch.randn(1, 64), i)
    dec.reset()
    fresh = DynamicCache()
    for i in range(3):
        x = torch.randn(1, 64)
        ref = m(inputs_embeds=x[None], past_key_values=fresh, use_cache=True,
                position_ids=torch.tensor([[i]]), return_dict=True)
        h, _ = dec.step(x, i)
        torch.testing.assert_close(h, ref.last_hidden_state[:, -1], atol=1e-4, rtol=1e-4)
    assert dec.n == 3
