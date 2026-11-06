"""Own-voice token embedder with user-feature fusion (the checkpoint's ``audio_tower``).

Each frame the backbone receives one vector that summarizes what the model said in the
previous frame (its ``GROUP_SIZE`` codec tokens) and what it heard (the ear's features of the
same frame):

    own  = output_matching(mean_g embed_tokens(audio_ids[g]))
    user = continual_output_matching(mean_g user_features[g])
    out  = own + user                      (continuous_features_mode == "add")

``embed_tokens`` is tied with the depth decoder's output head.
"""

from __future__ import annotations

import torch
from torch import nn

from awaken_chat.models.configuration import AwakenAudioConfig


class AudioTokenEmbedder(nn.Module):
    def __init__(self, config: AwakenAudioConfig) -> None:
        super().__init__()
        self.group_size = config.group_size
        self.hidden_size = config.output_dim
        self.continuous_features_mode = config.continuous_features_mode
        self.embed_tokens = nn.Embedding(config.codebook_size, self.hidden_size, config.pad_token_id)
        self.output_matching = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.continual_output_matching = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, audio_ids: torch.Tensor, user_features: torch.Tensor | None) -> torch.Tensor:
        """``audio_ids`` (B, G) and ``user_features`` (B, G, D) -> fused frame vector (B, D)."""
        own = self.output_matching(self.embed_tokens(audio_ids).mean(dim=1))
        if user_features is None:
            return own
        user = self.continual_output_matching(user_features.to(own.dtype).mean(dim=1))
        if self.continuous_features_mode == "add":
            return own + user
        return user
