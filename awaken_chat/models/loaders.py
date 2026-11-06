"""Loading the model and its text / mel front-ends from a local directory or the Hub."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase, WhisperFeatureExtractor

from awaken_chat.models.configuration import AwakenConfig
from awaken_chat.models.model import AwakenModel

logger = logging.getLogger(__name__)

MODEL_FILES = ["*.json", "*.safetensors", "*.txt", "*.jinja"]


def resolve_dir(repo_or_dir: str, allow_patterns: list[str] | None = None) -> str:
    """A local directory as-is, otherwise a Hub repo id downloaded to the cache."""
    if os.path.isdir(repo_or_dir):
        return repo_or_dir
    return snapshot_download(repo_or_dir, allow_patterns=allow_patterns)


@dataclass
class LoadedModel:
    model: AwakenModel
    tokenizer: PreTrainedTokenizerBase
    feature_extractor: WhisperFeatureExtractor
    path: str

    @property
    def device(self) -> torch.device:
        return self.model.device

    def prompt_ids(self, prompt: str) -> torch.Tensor:
        """Tokenize a system prompt and close it with the text BOS, shape (1, L)."""
        from awaken_chat.frame import TEXT

        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        ids = torch.cat([ids, torch.tensor([[TEXT.bos]])], dim=1)
        return ids.to(self.device)


def load_model(repo_or_dir: str, device: str | torch.device = "cuda",
               dtype: torch.dtype = torch.bfloat16,
               attn_implementation: str = "sdpa") -> LoadedModel:
    """Load weights, tokenizer and the Whisper feature extractor of the ear.

    Checkpoints trained with the Fun-Audio-Chat layout load unchanged; the config's
    ``continuous_features_mode`` is forced to ``"add"``, the mode every duplex checkpoint was
    trained with.
    """
    path = resolve_dir(repo_or_dir, MODEL_FILES)
    config = AwakenConfig.from_pretrained(path)
    config.audio_config.continuous_features_mode = "add"
    logger.info("loading model from %s (%s, %s)", path, dtype, attn_implementation)
    model = AwakenModel.from_pretrained(
        path, config=config, torch_dtype=dtype, attn_implementation=attn_implementation
    )
    model = model.to(device).eval()
    model.audio_tower.continuous_features_mode = "add"
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()  # return the fp32 -> bf16 conversion blocks to the driver
    tokenizer = AutoTokenizer.from_pretrained(path)
    if os.path.isfile(os.path.join(path, "preprocessor_config.json")):
        feature_extractor = WhisperFeatureExtractor.from_pretrained(path)
    else:  # trainer checkpoints carry no preprocessor config; the ear's mel recipe is fixed
        feature_extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000,
                                                    chunk_length=300)
    return LoadedModel(model, tokenizer, feature_extractor, path)
