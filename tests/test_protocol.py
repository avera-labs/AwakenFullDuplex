import json
import struct

import numpy as np

from awaken_chat.frame import AUDIO, TEXT
from awaken_chat.server.protocol import (
    AUDIO_FRAME,
    MIC_PCM,
    SILENT_FRAME,
    StartRequest,
    audio_class,
    decode_mic,
    encode_audio_frame,
    text_class,
    text_message,
)


def test_mic_roundtrip():
    pcm = np.array([0.0, 0.5, -0.5, 1.0], np.float32)
    payload = bytes([MIC_PCM]) + (pcm * 32767).astype("<i2").tobytes()
    out = decode_mic(payload)
    assert np.allclose(out, pcm, atol=1 / 32767)


def test_audio_frame_encoding():
    pcm = np.zeros(4800, np.float32)
    msg = encode_audio_frame(7, pcm)
    kind, frame = struct.unpack_from("<BI", msg)
    assert (kind, frame, len(msg)) == (AUDIO_FRAME, 7, 5 + 2 * 4800)
    silent = encode_audio_frame(8, None)
    assert struct.unpack("<BI", silent) == (SILENT_FRAME, 8)


def test_start_request_sanitizes():
    req = StartRequest.parse({"prompt": "  be   brief ", "seed": 12.0, "decode": "nope", "voice": "x" * 100})
    assert req.prompt == "be brief" and req.seed == 12 and req.decode == {} and len(req.voice) == 64
    assert StartRequest.parse({}).seed is None


def test_classes_and_messages():
    assert text_class(1000) == "txt" and text_class(TEXT.pad) == "pad" and text_class(TEXT.sil) == "sil"
    assert audio_class(AUDIO.bos) == "bos" and audio_class(AUDIO.eos) == "eos"
    assert audio_class(AUDIO.pad) == "pad" and audio_class(243) == "sil" and audio_class(5) == "voi"
    assert json.loads(text_message("ready", seed=3)) == {"type": "ready", "seed": 3}
