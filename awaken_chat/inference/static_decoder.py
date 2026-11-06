"""Single-token decode over a static KV cache for a Qwen3 stack, captured as one CUDA graph.

Both Qwen3 stacks of the model (the 8B backbone and the depth decoder) are driven one token
at a time. The stock transformers forward with a growing ``DynamicCache`` launches about two
thousand small kernels per step; this module fuses each layer into a few compiled kernels
(RMSNorm -> fused QKV -> q/k norm -> RoPE, attention over the cache, o_proj -> RMSNorm ->
fused gate/up -> SwiGLU -> down) and replays the whole step as a single CUDA graph.

Math is identical to ``transformers``' Qwen3 (bf16 noise only). Attention uses flash-attn's
``flash_attn_with_kvcache`` when installed and a masked SDPA over the static cache otherwise.

Cache layout: ``(layers, max_len, kv_heads, head_dim)``. Each step writes column ``t`` and
attends to columns ``[lo, t]``. ``lo`` is 0 for the backbone (prompt + every frame) and is
reset together with ``t`` for the depth decoder at every new utterance (block-local
attention and positions).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:  # pragma: no cover
    flash_attn_with_kvcache = None


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-b, a), dim=-1)


def _pre(h, ln_w, qkv_w, qn_w, kn_w, cos, sin, n_q: int, n_kv: int, hd: int, eps: float):
    """``h`` (1, D) -> q (1, 1, n_q, hd), k (1, 1, n_kv, hd), v (1, 1, n_kv, hd)."""
    a = _rms(h, ln_w, eps)
    qkv = F.linear(a, qkv_w)
    q, k, v = qkv.split([n_q * hd, n_kv * hd, n_kv * hd], dim=-1)
    q = _rms(q.reshape(1, n_q, hd), qn_w, eps)
    k = _rms(k.reshape(1, n_kv, hd), kn_w, eps)
    c, s = cos[:, None], sin[:, None]
    q = q * c + _rotate_half(q) * s
    k = k * c + _rotate_half(k) * s
    return q[:, None], k[:, None], v.reshape(1, 1, n_kv, hd)


def _post(h, o, o_w, ln2_w, gu_w, down_w, inter: int, eps: float):
    """Attention output ``o`` (1, 1, n_q, hd) -> next hidden (1, D)."""
    h = h + F.linear(o.reshape(1, -1), o_w)
    a = _rms(h, ln2_w, eps)
    g, u = F.linear(a, gu_w).split([inter, inter], dim=-1)
    return h + F.linear(F.silu(g) * u, down_w)


_COMPILED: dict[str, Callable] = {}


def _compiled(name: str, fn: Callable, enabled: bool) -> Callable:
    if not enabled:
        return fn
    if name not in _COMPILED:
        # One compile thread: Inductor's default worker pool creates one CUDA context per
        # worker at import, which costs minutes on a busy GPU; our two graphs compile in
        # seconds in-process.
        import torch._inductor.config as inductor_config

        inductor_config.compile_threads = 1
        _COMPILED[name] = torch.compile(fn, dynamic=False, fullgraph=True)
    return _COMPILED[name]


class StaticDecoder:
    """One Qwen3Model (``layers`` / ``norm`` / ``rotary_emb``) + optional head, batch 1."""

    def __init__(self, qwen3, max_len: int, head: Callable | None = None,
                 use_graph: bool = True, compile_on: bool = True) -> None:
        self.m, self.max_len, self.head, self.use_graph = qwen3, max_len, head, use_graph
        cfg = qwen3.config
        p = next(qwen3.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.n_layers = len(qwen3.layers)
        a0 = qwen3.layers[0].self_attn
        self.hd, self.scale, self.eps = a0.head_dim, a0.scaling, cfg.rms_norm_eps
        self.n_q = a0.q_proj.out_features // self.hd
        self.n_kv = a0.k_proj.out_features // self.hd
        self.dim, self.inter = cfg.hidden_size, cfg.intermediate_size
        if a0.q_proj.bias is not None or qwen3.layers[0].mlp.gate_proj.bias is not None:
            raise ValueError("StaticDecoder expects bias-free Qwen3 projections")
        shape = (self.n_layers, max_len, self.n_kv, self.hd)
        self.k = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.v = torch.zeros_like(self.k)
        self.qkv_w = torch.empty(self.n_layers, (self.n_q + 2 * self.n_kv) * self.hd, self.dim,
                                 dtype=self.dtype, device=self.device)
        self.gu_w = torch.empty(self.n_layers, 2 * self.inter, self.dim,
                                dtype=self.dtype, device=self.device)
        self.refresh_weights()
        self.lo = torch.zeros(1, dtype=torch.int32, device=self.device)  # window start
        self.t = torch.zeros(1, dtype=torch.int32, device=self.device)  # write column
        self.n = 0  # python mirror of t
        self.x = torch.zeros(1, self.dim, dtype=self.dtype, device=self.device)
        self.pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self.pre = _compiled("pre", _pre, compile_on)
        self.post = _compiled("post", _post, compile_on)
        self.h_out = self.logits_out = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.columns = torch.arange(max_len, device=self.device, dtype=torch.int32)

    @torch.no_grad()
    def refresh_weights(self) -> None:
        """Copy the HF weights into the fused QKV / gate-up buffers."""
        for li, layer in enumerate(self.m.layers):
            at, ml = layer.self_attn, layer.mlp
            torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], 0, out=self.qkv_w[li])
            torch.cat([ml.gate_proj.weight, ml.up_proj.weight], 0, out=self.gu_w[li])

    # State ------------------------------------------------------------------------------
    def reset(self) -> None:
        """Empty the cache: the next write goes to column 0 (block-local restart)."""
        self.lo.zero_()
        self.t.zero_()
        self.n = 0

    def load_prefix(self, cache, length: int) -> None:
        """Copy a transformers prefill cache (batch 1) into columns ``[0, length)``."""
        if length > self.max_len:
            raise ValueError(f"prefix {length} exceeds the static cache ({self.max_len})")
        for li in range(self.n_layers):
            if hasattr(cache, "layers"):
                k, v = cache.layers[li].keys, cache.layers[li].values
            else:
                k, v = cache.key_cache[li], cache.value_cache[li]
            self.k[li, :length] = k[0, :, :length].transpose(0, 1).to(self.dtype)
            self.v[li, :length] = v[0, :, :length].transpose(0, 1).to(self.dtype)
        self.lo.zero_()
        self.t.fill_(length)
        self.n = length

    @property
    def remaining(self) -> int:
        return self.max_len - self.n

    # Step -------------------------------------------------------------------------------
    def _attend(self, li: int, q, k, v):
        if flash_attn_with_kvcache is not None and q.is_cuda:
            return flash_attn_with_kvcache(
                q, self.k[li][None], self.v[li][None], k=k, v=v, cache_seqlens=self.t,
                cache_leftpad=self.lo, softmax_scale=self.scale, causal=False,
            )
        # Fallback: write column t, then masked SDPA over the whole static cache.
        col = self.t.long()
        self.k[li].index_copy_(0, col, k[0])
        self.v[li].index_copy_(0, col, v[0])
        valid = (self.columns >= self.lo) & (self.columns <= self.t)  # (max_len,)
        rep = self.n_q // self.n_kv
        kk = self.k[li].transpose(0, 1).repeat_interleave(rep, 0)[None]  # (1, n_q, L, hd)
        vv = self.v[li].transpose(0, 1).repeat_interleave(rep, 0)[None]
        qq = q.reshape(1, self.n_q, 1, self.hd)
        o = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=valid[None, None, None, :],
                                           scale=self.scale)
        return o.reshape(1, 1, self.n_q, self.hd)

    def _impl(self, x, pos):
        m = self.m
        cos, sin = m.rotary_emb(x[:, None], pos[:, None])
        cos, sin = cos[:, 0], sin[:, 0]
        h = x
        for li, layer in enumerate(m.layers):
            at = layer.self_attn
            q, k, v = self.pre(h, layer.input_layernorm.weight, self.qkv_w[li], at.q_norm.weight,
                               at.k_norm.weight, cos, sin, self.n_q, self.n_kv, self.hd, self.eps)
            o = self._attend(li, q, k, v)
            h = self.post(h, o, at.o_proj.weight, layer.post_attention_layernorm.weight,
                          self.gu_w[li], layer.mlp.down_proj.weight, self.inter, self.eps)
        h = _rms(h, m.norm.weight, self.eps)
        return h, (self.head(h) if self.head is not None else None)

    def _capture(self) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._impl(self.x, self.pos)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.h_out, self.logits_out = self._impl(self.x, self.pos)

    @torch.no_grad()
    def step(self, x: torch.Tensor, position: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One token: ``x`` (1, D) input embedding at RoPE ``position`` -> (normed hidden (1, D),
        head logits or None). Outputs are the graph's static buffers: clone them before the
        next step if they must survive."""
        if self.n >= self.max_len:
            raise RuntimeError(f"static cache full ({self.max_len})")
        self.x.copy_(x.to(self.dtype))
        self.pos.fill_(position)
        if self.use_graph:
            if self.graph is None:
                self._capture()
            self.graph.replay()
            h, logits = self.h_out, self.logits_out
        else:
            h, logits = self._impl(self.x, self.pos)
        self.t += 1
        self.n += 1
        return h, logits
