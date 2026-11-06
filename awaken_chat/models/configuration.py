"""Model configuration.

``AwakenConfig`` is the checkpoint's ``config.json``: a text backbone config (Qwen3) plus an
``AwakenAudioConfig`` describing the three audio modules (ear, token embedder, depth decoder).

The layout is inherited from Fun-Audio-Chat (Apache-2.0), whose checkpoints load unchanged:
the key names and sub-config structure are identical, only training-time fields were removed.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING

AUDIO_MODEL_TYPE = "awaken_chat_audio"
MODEL_TYPE = "awaken_chat"


class AwakenAudioConfig(PretrainedConfig):
    """Configuration of the audio modules.

    Ear (``continuous_audio_tower``, a Whisper-style encoder made causal):
        num_mel_bins, encoder_layers, encoder_attention_heads, encoder_ffn_dim, d_model,
        activation_function, max_source_positions, output_dim (= the backbone hidden size).
    Token embedder (``audio_tower``):
        codebook_size (6561 codes + bos/eos/pad), group_size (codec tokens per frame),
        continuous_features_mode ("add": user features are added to the own-voice embedding).
    Depth decoder (``audio_invert_tower``):
        crq_transformer_config (a small Qwen3 config) and the codebook_size shared head.
    """

    model_type = AUDIO_MODEL_TYPE

    def __init__(
        self,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        d_model: int = 1280,
        activation_function: str = "gelu",
        max_source_positions: int = 1500,
        output_dim: int = 4096,
        codebook_size: int = 6564,
        group_size: int = 5,
        continuous_features_mode: str = "add",
        crq_transformer_config: dict[str, Any] | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        # Training-only fields present in older configs are accepted and dropped.
        for key in ("dropout", "attention_dropout", "activation_dropout", "scale_embedding",
                    "initializer_range", "n_window", "enable_audio_invert_tower",
                    "num_hidden_layers", "encoder_layerdrop"):
            kwargs.pop(key, None)
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         pad_token_id=pad_token_id, **kwargs)
        self.num_mel_bins = num_mel_bins
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.d_model = d_model
        self.activation_function = activation_function
        self.max_source_positions = max_source_positions
        self.output_dim = output_dim
        self.codebook_size = codebook_size
        self.group_size = group_size
        self.continuous_features_mode = continuous_features_mode
        self.crq_transformer_config = crq_transformer_config


class AwakenConfig(PretrainedConfig):
    """Top-level configuration: ``text_config`` (Qwen3) + ``audio_config``."""

    model_type = MODEL_TYPE
    sub_configs = {"text_config": PretrainedConfig, "audio_config": AwakenAudioConfig}

    def __init__(
        self,
        audio_config: dict[str, Any] | AwakenAudioConfig | None = None,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        hidden_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        for key in ("audio_token_index", "ignore_index"):
            kwargs.pop(key, None)
        if isinstance(audio_config, dict):
            audio_config = dict(audio_config)
            audio_config.pop("model_type", None)
            audio_config = AwakenAudioConfig(**audio_config)
        elif audio_config is None:
            audio_config = AwakenAudioConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_model_type = text_config.pop("model_type", "qwen3")
            text_config = CONFIG_MAPPING[text_model_type](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen3"]()
        self.text_config = text_config
        self.hidden_size = hidden_size if hidden_size is not None else text_config.hidden_size
        super().__init__(**kwargs)


__all__ = ["AwakenAudioConfig", "AwakenConfig", "AUDIO_MODEL_TYPE", "MODEL_TYPE"]
