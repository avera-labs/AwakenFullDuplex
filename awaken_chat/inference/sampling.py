"""Sampling helpers shared by the text lane and the depth decoder."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def sample_top_k(logits: torch.Tensor, temperature: float, k: int) -> torch.Tensor:
    """Top-k / temperature sample, ``(B, V)`` -> ``(B,)``. ``temperature == 0`` is argmax."""
    logits = logits.float()
    if temperature == 0.0:
        return logits.argmax(-1)
    if k <= 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    values, index = logits.topk(min(k, logits.shape[-1]), dim=-1)
    probs = torch.softmax(values / temperature, dim=-1)
    choice = torch.multinomial(probs, num_samples=1)
    return index.gather(-1, choice).squeeze(-1)


def class_mask(ids: Iterable[int], vocab: int, device: torch.device) -> torch.Tensor:
    """Boolean ``(vocab,)`` vector that is True at the given ids."""
    mask = torch.zeros(vocab, dtype=torch.bool, device=device)
    keep = [i for i in ids if i < vocab]
    if keep:
        mask[torch.tensor(keep, device=device)] = True
    return mask


def apply_repetition_penalty(logits: torch.Tensor, recent_ids: Iterable[int],
                             penalty: float) -> torch.Tensor:
    """HF-style penalty on recently generated ids: positive logits are divided, negative
    ones multiplied. Returns a new tensor; ``penalty <= 1`` is a no-op."""
    ids = sorted(set(recent_ids))
    if penalty <= 1.0 or not ids:
        return logits
    logits = logits.clone()
    index = torch.tensor(ids, device=logits.device)
    values = logits[0, index]
    logits[0, index] = torch.where(values > 0, values / penalty, values * penalty)
    return logits
