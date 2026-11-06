"""Decode parameters and session limits.

Two presets are shipped. ``bench`` is the recipe every reported benchmark number was produced
with. ``live`` is the sharper recipe the demo server runs for cleaner speech in a
conversation (near-greedy inside a text burst, a repetition penalty on content tokens and a
cap on consecutive near-silent codec tokens).
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DecodeParams:
    # Text lane. ``*_burst`` apply inside a burst (previous token was content); <= 0 means
    # "same as the idle values". Onset (listening -> speaking) is a sampling decision, so the
    # idle values should not be near-greedy.
    text_temperature: float = 0.7
    text_top_k: int = 25
    text_temperature_burst: float = -1.0
    text_top_k_burst: int = -1
    # Penalty on the last ``text_rep_window`` generated content ids (1.0 = off). Control ids
    # are never penalized, so onset / release behaviour is untouched.
    text_rep_penalty: float = 1.0
    text_rep_window: int = 64
    # Depth decoder.
    audio_temperature: float = 0.8
    audio_top_k: int = 250
    # Mask the near-silent codec cluster after this many in a row inside an utterance
    # (-1 = off). Counters the "text flows, audio stays silent" attractor.
    max_silent_run: int = -1
    # Hard cap on one utterance, in seconds; on hitting it the pending audio frame is clamped
    # to EOS and the text lane releases, like the data's own endings.
    max_utterance_s: float = 20.0

    @property
    def burst_temperature(self) -> float:
        return self.text_temperature_burst if self.text_temperature_burst > 0 else self.text_temperature

    @property
    def burst_top_k(self) -> int:
        return self.text_top_k_burst if self.text_top_k_burst > 0 else self.text_top_k

    def with_overrides(self, **changes: float | int) -> DecodeParams:
        return replace(self, **changes)


BENCH = DecodeParams()
LIVE = DecodeParams(
    text_temperature=0.5, text_top_k=10,
    text_temperature_burst=0.2, text_top_k_burst=5,
    text_rep_penalty=1.15, audio_temperature=0.8, audio_top_k=250, max_silent_run=3,
    # Let spoken words catch up with the text lane, within the default ~41 s depth budget.
    max_utterance_s=40.0,
)
PRESETS = {"bench": BENCH, "live": LIVE}


@dataclass(frozen=True)
class SessionLimits:
    """Static cache sizes. ``backbone_len`` bounds prompt + one position per frame of the whole
    conversation (6144 = 20 min at 5 Hz with a short prompt); ``depth_len`` bounds one
    utterance (5 positions per frame; 1024 = 41 s, above ``max_utterance_s``)."""

    backbone_len: int = 6144
    depth_len: int = 1024
