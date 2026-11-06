"""Depth decoder (the checkpoint's ``audio_invert_tower``): one backbone state -> 5 codec tokens.

For every voiced frame the backbone's final hidden state (plus the text embedding of the
token fed on that frame) is expanded into ``GROUP_SIZE`` slots. A small causal transformer
("CRQ", a Qwen3 stack) then predicts the codec tokens of the frame one at a time, each slot
also receiving the embedding of the token sampled in the previous slot:

    slots      = pre_matching(h + text_emb).view(G, D)
    slots[0]  += embed(last token of the previous frame)
    for g in 0..G-1:
        logits_g   = lm_head(output_matching(crq(input_matching(slots[g]))))
        tok_g      = sample(logits_g)
        slots[g+1] += embed(tok_g)

Attention is block-local: the CRQ cache is reset at the start of every utterance and its
positions restart from 0, so an utterance never attends to the previous one.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from awaken_chat.models.configuration import AwakenAudioConfig


class DepthDecoder(nn.Module):
    def __init__(self, config: AwakenAudioConfig) -> None:
        super().__init__()
        self.group_size = config.group_size
        self.hidden_size = config.output_dim
        self.pre_matching = nn.Linear(self.hidden_size, self.hidden_size * self.group_size)
        crq_config = AutoConfig.for_model(**config.crq_transformer_config)
        self.crq_transformer = AutoModel.from_config(crq_config)
        del self.crq_transformer.embed_tokens
        self.input_matching = nn.Linear(self.hidden_size, crq_config.hidden_size, bias=False)
        self.output_matching = nn.Linear(crq_config.hidden_size, self.hidden_size, bias=False)
        self.lm_head = nn.Linear(self.hidden_size, config.codebook_size, bias=False)

    def embed(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Codec token ids -> embeddings (the head's rows; tied with the token embedder)."""
        return self.lm_head.weight.data[audio_tokens]

    def expand(self, backbone_hidden: torch.Tensor, text_embedding: torch.Tensor,
               last_token: torch.Tensor) -> torch.Tensor:
        """Frame input -> the ``(B, G, D)`` slot tensor, slot 0 seeded with ``last_token``."""
        slots = self.pre_matching(backbone_hidden + text_embedding)
        slots = slots.reshape(slots.shape[0], self.group_size, self.hidden_size).clone()
        slots[:, 0] += self.embed(last_token)
        return slots

    def head(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """The CRQ hidden -> codec logits map, for the static decoder's graph."""
        return lambda h: self.lm_head(self.output_matching(h))
