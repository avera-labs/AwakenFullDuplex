"""Frame specification: the single source of truth for every constant of the duplex stream.

Awaken Chat runs at a fixed cadence. Every 200 ms ("one frame") the model consumes one text
token and one group of ``GROUP_SIZE`` codec tokens of its own voice, plus the user-audio
features of the same 200 ms, and predicts the next text token and the next codec group.

Time grid (one frame):

====================  =========  =================================================
quantity              value      note
====================  =========  =================================================
FRAME_HZ              5          frames per second
GROUP_SIZE            5          codec tokens per frame (40 ms each)
SR_IN                 24000      user audio and generated speech sample rate
SR_MEL                16000      the ear's mel front-end runs on 16 kHz audio
MEL_PER_FRAME         20         mel columns (10 ms hop) per frame
EAR_TOKENS_PER_FRAME  10         post-conv tower tokens per frame (conv stride 2)
EAR_POS_PERIOD        100        the tower's positional embedding repeats mod 100
EAR_ATTN_WINDOW       100        causal attention window in tower tokens (2 s)
====================  =========  =================================================
"""

from __future__ import annotations

from dataclasses import dataclass

FRAME_HZ = 5
GROUP_SIZE = 5
SR_IN = 24_000
SR_MEL = 16_000
SAMPLES_PER_FRAME_IN = SR_IN // FRAME_HZ  # 4800
SAMPLES_PER_FRAME_MEL = SR_MEL // FRAME_HZ  # 3200
MEL_BINS = 128
MEL_HOP = 160  # 10 ms at 16 kHz
MEL_PER_FRAME = 20
EAR_TOKENS_PER_FRAME = MEL_PER_FRAME // 2
EAR_POS_PERIOD = 100
EAR_ATTN_WINDOW = 100
CODEC_SAMPLES_PER_TOKEN = SR_IN // (FRAME_HZ * GROUP_SIZE)  # 960


@dataclass(frozen=True)
class TextTokens:
    """Special ids of the text lane (Qwen3 tokenizer extended by Fun-Audio-Chat).

    Every id below ``first_special`` is a content token. The four control ids mean:

    * ``bos``  ``<|audio_bos|>`` closes the system prompt (the record head).
    * ``eos``  ``<|audio_eos|>`` legacy end marker, never sampled by current checkpoints.
    * ``pad``  ``<|audio_pad|>`` the model is *listening* (talker off). Sampled while the
      model is voicing it is the *release*: the utterance ends on that frame.
    * ``sil``  ``<|audio_sil|>`` words are done but the audio is still voicing.
    """

    bos: int = 151670
    eos: int = 151671
    pad: int = 151672
    sil: int = 151673
    first_special: int = 151643

    def is_content(self, token_id: int) -> bool:
        return token_id < self.first_special


@dataclass(frozen=True)
class AudioTokens:
    """Special ids of the codec lane (CosyVoice3 FSQ codebook of 6561 codes + 3 controls).

    * ``bos`` fills the first frame of an utterance (fed once, never predicted).
    * ``eos`` fills the release frame (fed once, then the session goes idle).
    * ``pad`` fills every idle frame.
    * ``quiet`` is the empirical near-silence cluster of the codebook (frame RMS <= 0.003),
      used by the optional consecutive-silence cap of the depth decoder.
    """

    bos: int = 6561
    eos: int = 6562
    pad: int = 6563
    codebook_size: int = 6561
    quiet: frozenset[int] = frozenset((28, 243, 244, 245, 271, 2430))

    def is_code(self, token_id: int) -> bool:
        return token_id < self.codebook_size


TEXT = TextTokens()
AUDIO = AudioTokens()

SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "You are provided with user audio while you speak. "
    "You are asked to generate both text and speech tokens at the same time."
    "<|im_end|>\n<|im_start|>duo\n"
)


def system_prompt(extra: str = "") -> str:
    """The system prompt, optionally extended with a persona / instruction sentence."""
    extra = " ".join(extra.split())
    if not extra:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT.replace("<|im_end|>", " " + extra + "<|im_end|>", 1)
