"""Awaken Chat: a full-duplex speech-to-speech conversation model (inference package).

Quick start::

    from awaken_chat import load_model, Decoders, DuplexSession, system_prompt

    loaded = load_model("<checkpoint dir or Hub repo>")
    decoders = Decoders(loaded.model)
    session = DuplexSession(loaded.model, decoders, loaded.prompt_ids(system_prompt()))
    # every 200 ms: feats = ear features of the frame -> session.step(feats)

See ``awaken-server`` for live use and ``awaken-offline`` for a recorded track.
"""

from awaken_chat.frame import AUDIO, SYSTEM_PROMPT, TEXT, system_prompt
from awaken_chat.inference.params import BENCH, LIVE, DecodeParams, SessionLimits
from awaken_chat.inference.session import Decoders, DuplexSession, FrameOut
from awaken_chat.models.configuration import AwakenAudioConfig, AwakenConfig
from awaken_chat.models.loaders import LoadedModel, load_model
from awaken_chat.models.model import AwakenModel

__version__ = "0.1.0"


def register() -> None:
    """Register the config and model with ``transformers``' Auto classes."""
    from transformers import AutoConfig, AutoModel

    AutoConfig.register(AwakenConfig.model_type, AwakenConfig)
    AutoConfig.register(AwakenAudioConfig.model_type, AwakenAudioConfig)
    AutoModel.register(AwakenConfig, AwakenModel)


__all__ = [
    "AUDIO", "TEXT", "SYSTEM_PROMPT", "system_prompt",
    "BENCH", "LIVE", "DecodeParams", "SessionLimits",
    "Decoders", "DuplexSession", "FrameOut",
    "AwakenAudioConfig", "AwakenConfig", "AwakenModel",
    "LoadedModel", "load_model", "register", "__version__",
]
