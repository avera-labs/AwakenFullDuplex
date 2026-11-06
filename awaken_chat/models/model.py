"""The full-duplex model: ear + token embedder + Qwen3 backbone + depth decoder.

Checkpoint layout (``model.safetensors`` keys):

    continuous_audio_tower.*   CausalAudioEncoder   (the ear)
    audio_tower.*              AudioTokenEmbedder   (own-voice tokens + user features -> frame)
    language_model.*           Qwen3ForCausalLM     (text lane; 8B)
    audio_invert_tower.*       DepthDecoder         (frame state -> 5 codec tokens)

This module holds the weights and the per-frame input assembly. The decoding loop lives in
:mod:`awaken_chat.inference.session`.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.cache_utils import DynamicCache

from awaken_chat.models.configuration import AwakenConfig
from awaken_chat.models.depth import DepthDecoder
from awaken_chat.models.ear import CausalAudioEncoder
from awaken_chat.models.fusion import AudioTokenEmbedder


class AwakenModel(PreTrainedModel):
    config_class = AwakenConfig
    base_model_prefix = "model"
    _no_split_modules = ["EarLayer", "Qwen3DecoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def __init__(self, config: AwakenConfig) -> None:
        super().__init__(config)
        self.continuous_audio_tower = CausalAudioEncoder(config.audio_config)
        self.audio_tower = AudioTokenEmbedder(config.audio_config)
        self.audio_invert_tower = DepthDecoder(config.audio_config)
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.post_init()

    # PreTrainedModel plumbing ---------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        # Inference-only: weights always come from a checkpoint.
        pass

    def tie_weights(self) -> None:
        self._tie_or_clone_weights(self.audio_invert_tower.lm_head, self.audio_tower.embed_tokens)
        self.language_model.tie_weights()

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    # Convenience views ----------------------------------------------------------------
    @property
    def ear(self) -> CausalAudioEncoder:
        return self.continuous_audio_tower

    @property
    def embedder(self) -> AudioTokenEmbedder:
        return self.audio_tower

    @property
    def depth(self) -> DepthDecoder:
        return self.audio_invert_tower

    @property
    def group_size(self) -> int:
        return self.config.audio_config.group_size

    # Per-frame input assembly ---------------------------------------------------------
    def embed_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings()(token_ids)

    def frame_input(self, text_embedding: torch.Tensor, audio_ids: torch.Tensor,
                    user_features: torch.Tensor | None) -> torch.Tensor:
        """The backbone input of one frame: ``(text_emb + fused_audio) / 2``, shape (B, D)."""
        fused = self.audio_tower(audio_ids, user_features)
        return (text_embedding + fused) / 2

    @torch.no_grad()
    def prefill(self, prompt_ids: torch.Tensor) -> DynamicCache:
        """Run the system prompt through the backbone; returns its KV cache."""
        out = self.language_model(
            inputs_embeds=self.embed_text(prompt_ids), use_cache=True, return_dict=True
        )
        return out.past_key_values


__all__ = ["AwakenModel"]
