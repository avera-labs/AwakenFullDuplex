"""WebSocket protocol of the live server.

One WebSocket connection is one conversation. Messages are JSON text frames in both directions
except microphone audio and generated speech, which are binary frames.

Client -> server
    text  {"type": "start", "prompt": str?, "voice": str?, "preset": "live"|"bench"?,
           "decode": {DecodeParams fields}?, "seed": int?}
    binary 0x01 + int16 little-endian PCM at 24 kHz, mono (any chunk size)
    text  {"type": "ping"}                      keep-alive
    text  {"type": "stop"}

Server -> client
    text  {"type": "status", "message": str}    while loading / warming up
    text  {"type": "ready", "seed": int, "decode": {...}}
    binary 0x02 + uint32 LE frame index + int16 LE PCM (4800 samples, one 200 ms frame),
           or 0x03 + uint32 LE frame index for an all-silent frame (no payload)
    text  {"type": "frames", "items": [{"f": int, "t": float, "text": str|null,
           "text_class": "txt"|"pad"|"sil"|"bos"|"eos", "audio_class": [str x 5]}]}
    text  {"type": "utterance", "text": str, "t0": float, "t1": float}
    text  {"type": "stats", ...}
    text  {"type": "error", "message": str}
    text  {"type": "end", "frames": int}

Frame indices let the client keep one steady 5 Hz playback clock: the server emits exactly one
audio frame (speech or silence) per generated frame.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from awaken_chat.frame import AUDIO, TEXT

MIC_PCM = 0x01
AUDIO_FRAME = 0x02
SILENT_FRAME = 0x03


@dataclass
class StartRequest:
    prompt: str = ""
    voice: str = ""
    preset: str = "live"
    decode: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None

    @classmethod
    def parse(cls, msg: dict[str, Any]) -> StartRequest:
        decode = msg.get("decode")
        seed = msg.get("seed")
        return cls(
            prompt=" ".join(str(msg.get("prompt") or "").split())[:2000],
            voice=str(msg.get("voice") or "")[:64],
            preset=str(msg.get("preset") or "live"),
            decode=decode if isinstance(decode, dict) else {},
            seed=int(seed) % (1 << 30) if isinstance(seed, (int, float)) else None,
        )


def text_message(kind: str, **fields: Any) -> str:
    return json.dumps({"type": kind, **fields})


def decode_mic(payload: bytes) -> np.ndarray:
    """Binary mic frame -> float32 samples in [-1, 1]."""
    if not payload or payload[0] != MIC_PCM:
        raise ValueError("expected a mic PCM frame")
    return np.frombuffer(payload[1:], dtype="<i2").astype(np.float32) / 32768.0


def encode_audio_frame(frame: int, pcm: np.ndarray | None) -> bytes:
    if pcm is None:
        return struct.pack("<BI", SILENT_FRAME, frame)
    data = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
    return struct.pack("<BI", AUDIO_FRAME, frame) + data


def text_class(token_id: int) -> str:
    if TEXT.is_content(token_id):
        return "txt"
    return {TEXT.bos: "bos", TEXT.eos: "eos", TEXT.pad: "pad", TEXT.sil: "sil"}.get(token_id, "spec")


def audio_class(token_id: int) -> str:
    if token_id == AUDIO.bos:
        return "bos"
    if token_id == AUDIO.eos:
        return "eos"
    if token_id == AUDIO.pad:
        return "pad"
    if token_id in AUDIO.quiet:
        return "sil"
    return "voi"


def params_dict(params) -> dict[str, Any]:
    return asdict(params)
