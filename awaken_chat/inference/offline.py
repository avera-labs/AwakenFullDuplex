"""Offline conversation: run a recorded user track through the model and render the reply.

The user track is encoded in one causal pass, the session is stepped frame by frame over the
whole span, and every voiced run of codec tokens is rendered to 24 kHz speech placed on the
frame grid. The result is a stereo track (left: user, right: model) plus the token history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from awaken_chat.frame import (
    AUDIO,
    CODEC_SAMPLES_PER_TOKEN,
    FRAME_HZ,
    SAMPLES_PER_FRAME_IN,
    SR_IN,
    SR_MEL,
    TEXT,
    system_prompt,
)
from awaken_chat.inference.params import BENCH, DecodeParams
from awaken_chat.inference.session import Decoders, DuplexSession, FrameOut
from awaken_chat.models.ear import mel_of_track
from awaken_chat.models.loaders import LoadedModel


@dataclass
class Utterance:
    start_s: float
    end_s: float
    text: str


@dataclass
class OfflineResult:
    frames: list[FrameOut]
    user_track: np.ndarray  # 24 kHz
    agent_track: np.ndarray  # 24 kHz, same length
    utterances: list[Utterance] = field(default_factory=list)

    @property
    def text_ids(self) -> list[int]:
        return [f.text_id for f in self.frames]

    @property
    def audio_frames(self) -> list[list[int]]:
        return [f.audio_ids for f in self.frames]

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        n = max(len(self.user_track), len(self.agent_track))
        stereo = np.zeros((n, 2), dtype=np.float32)
        stereo[: len(self.user_track), 0] = self.user_track
        stereo[: len(self.agent_track), 1] = self.agent_track
        sf.write(str(out_dir / "stereo.wav"), stereo, SR_IN)
        torch.save({"text_ids": torch.tensor(self.text_ids),
                    "audio_frames": torch.tensor(self.audio_frames)}, out_dir / "tokens.pt")
        lines = [f"[{u.start_s:.1f}-{u.end_s:.1f}] {u.text}" for u in self.utterances]
        (out_dir / "transcript.txt").write_text("\n".join(lines) + "\n")


def load_user_track(path: str | Path) -> np.ndarray:
    """Any audio file -> mono float32 at 24 kHz."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR_IN:
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR_IN).numpy()
    return wav


def encode_user_track(loaded: LoadedModel, track24: np.ndarray, n_frames: int) -> torch.Tensor:
    """User features ``(n_frames, GROUP_SIZE, D)`` of a 24 kHz track via the whole-track ear."""
    wav16 = torchaudio.functional.resample(torch.from_numpy(track24).float(), SR_IN, SR_MEL)
    mel = mel_of_track(wav16.numpy(), loaded.feature_extractor, n_frames)
    return loaded.model.ear.encode_track(mel)


def utterances_of(frames: list[FrameOut], tokenizer) -> list[Utterance]:
    """Group content text tokens into utterances delimited by the model's own releases."""
    out: list[Utterance] = []
    ids: list[int] = []
    start = 0
    voicing = False
    for f in frames:
        is_bos = f.audio_ids[0] == AUDIO.bos
        is_eos = f.audio_ids[0] == AUDIO.eos
        if is_bos:
            voicing, start, ids = True, f.frame, []
        if TEXT.is_content(f.text_id):
            ids.append(f.text_id)
        if voicing and is_eos:
            text = tokenizer.decode(ids).strip()
            if text:
                out.append(Utterance(start / FRAME_HZ, (f.frame + 1) / FRAME_HZ, text))
            voicing, ids = False, []
    if ids:
        out.append(Utterance(start / FRAME_HZ, len(frames) / FRAME_HZ, tokenizer.decode(ids).strip()))
    return out


def render_agent_track(audio_frames: list[list[int]], detokenizer, voice=None) -> np.ndarray:
    """Codec frames -> 24 kHz track: each voiced run decoded whole, placed on the frame grid."""
    flat = [t for frame in audio_frames for t in frame]
    is_code = [AUDIO.is_code(t) for t in flat]
    parts: list[np.ndarray] = []
    i, n = 0, len(flat)
    while i < n:
        j = i
        while j < n and is_code[j] == is_code[i]:
            j += 1
        expected = (j - i) * CODEC_SAMPLES_PER_TOKEN
        if is_code[i]:
            wav = detokenizer.decode_tokens(torch.tensor(flat[i:j]), voice=voice).numpy()
            if wav.shape[0] < expected:
                wav = np.concatenate([wav, np.zeros(expected - wav.shape[0], np.float32)])
            parts.append(wav[:expected])
        else:
            parts.append(np.zeros(expected, np.float32))
        i = j
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


@torch.no_grad()
def run_offline(loaded: LoadedModel, decoders: Decoders, user_track: np.ndarray,
                params: DecodeParams = BENCH, prompt_extra: str = "", seed: int = 0,
                tail_s: float = 5.0, detokenizer=None, voice=None) -> OfflineResult:
    """Step the model over ``user_track`` (24 kHz) plus ``tail_s`` of trailing silence."""
    torch.manual_seed(seed)
    n_frames = int(np.ceil(len(user_track) / SAMPLES_PER_FRAME_IN)) + int(tail_s * FRAME_HZ)
    n_frames = -(-n_frames // 10) * 10  # multiple of 10 frames (2 s, the ear's position period)
    track = np.zeros(n_frames * SAMPLES_PER_FRAME_IN, np.float32)
    track[: len(user_track)] = user_track
    feats = encode_user_track(loaded, track, n_frames)
    session = DuplexSession(loaded.model, decoders, loaded.prompt_ids(system_prompt(prompt_extra)),
                            params=params)
    frames = [session.step(feats[f]) for f in range(n_frames)]
    agent = np.zeros(0, np.float32)
    if detokenizer is not None:
        agent = render_agent_track([f.audio_ids for f in frames], detokenizer, voice)
    return OfflineResult(frames, track, agent, utterances_of(frames, loaded.tokenizer))
