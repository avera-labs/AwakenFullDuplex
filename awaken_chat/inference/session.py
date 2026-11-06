"""The duplex session: the per-frame state machine that drives the model.

One :class:`DuplexSession` is one conversation. After the system prompt is prefilled, every
call to :meth:`DuplexSession.step` consumes the user features of one 200 ms frame and
produces the model's own frame: the text token and the five codec tokens that are *fed* on
that frame, plus the text token sampled for the next one.

State machine (``text`` = token fed this frame, sampled on the previous one)::

    IDLE     audio frame fed = PAD x 5
             text is content           -> feed BOS x 5 instead, enter VOICING
    VOICING  audio frame fed = the codec tokens predicted on the previous frame
             sample next text:
               content                 -> keep talking
               SIL                     -> words done, audio continues (content masked from here)
               PAD                     -> the RELEASE: next audio frame = EOS x 5, then IDLE
             depth decoder predicts the next audio frame (skipped on the release frame)

Both Qwen3 stacks run through :class:`~awaken_chat.inference.static_decoder.StaticDecoder`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import torch

from awaken_chat.frame import AUDIO, FRAME_HZ, TEXT, AudioTokens, TextTokens
from awaken_chat.inference.params import BENCH, DecodeParams, SessionLimits
from awaken_chat.inference.sampling import apply_repetition_penalty, class_mask, sample_top_k
from awaken_chat.inference.static_decoder import StaticDecoder
from awaken_chat.models.model import AwakenModel


class State(str, Enum):
    IDLE = "idle"
    VOICING = "voicing"


@dataclass(frozen=True)
class FrameOut:
    """What the model did on one frame."""

    frame: int
    text_id: int  # text token fed on this frame
    audio_ids: list[int]  # codec tokens fed on this frame (PAD / BOS / codes / EOS)
    next_text_id: int  # text token sampled for the next frame
    state: State  # state after the frame


class Decoders:
    """The two static decoders of one loaded model, reused across sessions."""

    def __init__(self, model: AwakenModel, limits: SessionLimits | None = None,
                 use_graph: bool = True, compile_on: bool = True) -> None:
        limits = limits or SessionLimits()
        self.backbone = StaticDecoder(model.language_model.model, limits.backbone_len,
                                      head=model.language_model.lm_head,
                                      use_graph=use_graph, compile_on=compile_on)
        self.depth = StaticDecoder(model.depth.crq_transformer, limits.depth_len,
                                   head=model.depth.head(), use_graph=use_graph,
                                   compile_on=compile_on)


class DuplexSession:
    def __init__(self, model: AwakenModel, decoders: Decoders, prompt_ids: torch.Tensor,
                 params: DecodeParams = BENCH, text: TextTokens = TEXT,
                 audio: AudioTokens = AUDIO, utterance_cap: bool = True) -> None:
        """``prompt_ids`` (1, L): the tokenized system prompt ending with ``text.bos``.
        ``utterance_cap`` applies ``params.max_utterance_s`` inside the session; offline
        drivers that enforce their own cap pass False."""
        self.model, self.dec, self.params = model, decoders, params
        self.text_tokens, self.audio_tokens = text, audio
        self.device = prompt_ids.device
        self.G = model.group_size
        self.utterance_cap = utterance_cap

        cache = model.prefill(prompt_ids)
        self.dec.backbone.load_prefix(cache, prompt_ids.shape[1])
        self.dec.depth.reset()
        del cache

        self.position = prompt_ids.shape[1]  # backbone RoPE position of the next frame
        self.frame = 0
        self.state = State.IDLE
        self.next_text = torch.tensor([text.pad], device=self.device)  # frame 0: listening
        self.next_audio: torch.Tensor | None = None
        self.retire_after_feed = False
        self.depth_frame = 0  # frames decoded in the current utterance
        self.silent_run = 0
        self.recent_content: deque[int] = deque(maxlen=params.text_rep_window)
        self.text_history: list[int] = []
        self.audio_history: list[list[int]] = []
        vocab = model.depth.lm_head.weight.shape[0]
        self.quiet_mask = class_mask(audio.quiet, vocab, self.device)

    # Helpers --------------------------------------------------------------------------
    @property
    def next_text_id(self) -> int:
        return int(self.next_text.item())

    def _filled(self, token_id: int) -> torch.Tensor:
        return torch.full((1, self.G), token_id, device=self.device)

    def _release(self) -> None:
        """Clamp the pending audio frame to EOS; the next step feeds it and goes idle."""
        self.next_audio = self._filled(self.audio_tokens.eos)
        self.retire_after_feed = True

    def force_eos(self) -> None:
        """Hard cap: end the utterance now, text lane releases with PAD like the data."""
        if self.state is State.VOICING:
            self._release()
        self.next_text = torch.tensor([self.text_tokens.pad], device=self.device)

    # The step -------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, user_features: torch.Tensor) -> FrameOut:
        """Advance one frame. ``user_features``: the ear's ``(GROUP_SIZE, D)`` for this frame."""
        m, text, audio, p = self.model, self.text_tokens, self.audio_tokens, self.params
        text_in = self.next_text
        text_in_id = int(text_in.item())

        # 1. Which audio frame is fed, and the state transition it implies.
        if self.state is State.IDLE:
            if text.is_content(text_in_id):
                audio_in = self._filled(audio.bos)
                self.state = State.VOICING
                self.dec.depth.reset()
                self.depth_frame = 0
                self.silent_run = 0
            else:
                audio_in = self._filled(audio.pad)
        else:
            audio_in = self.next_audio
            if self.retire_after_feed:
                self.state = State.IDLE
                self.retire_after_feed = False
                self.dec.depth.reset()
        self.text_history.append(text_in_id)
        self.audio_history.append([int(x) for x in audio_in[0].tolist()])

        # 2. Backbone step.
        text_emb = m.embed_text(text_in)  # (1, D)
        x = m.frame_input(text_emb, audio_in, user_features[None].to(self.device))
        hidden, logits = self.dec.backbone.step(x, self.position)
        hidden, logits = hidden.clone(), logits.clone()
        self.position += 1

        # 3. Text lane.
        logits = apply_repetition_penalty(logits, self.recent_content, p.text_rep_penalty)
        in_burst = text.is_content(text_in_id)
        if self.state is State.VOICING and not in_burst:
            # Inside an utterance the lane never returns from SIL/PAD to content.
            logits = logits.clone()
            logits[:, : text.first_special] = float("-inf")
        if in_burst:
            self.next_text = sample_top_k(logits, p.burst_temperature, p.burst_top_k)
        else:
            self.next_text = sample_top_k(logits, p.text_temperature, p.text_top_k)
        next_id = int(self.next_text.item())
        if text.is_content(next_id):
            self.recent_content.append(next_id)

        # 4. Depth decoder: the next audio frame.
        if self.state is State.VOICING:
            if next_id == text.pad:
                self._release()  # no depth call on the release frame
            elif (self.depth_frame + 1) * self.G > self.dec.depth.max_len or (
                self.utterance_cap and self.depth_frame >= p.max_utterance_s * FRAME_HZ
            ):
                self.force_eos()
            else:
                self.next_audio = self._decode_audio_frame(hidden, text_emb, audio_in[:, -1])
                self.depth_frame += 1
                if int((self.next_audio == audio.eos).sum()) >= 1:
                    self._release()  # safety: a sampled EOS retires the utterance cleanly
        self.frame += 1
        return FrameOut(self.frame - 1, text_in_id, self.audio_history[-1], self.next_text_id,
                        self.state)

    def _decode_audio_frame(self, hidden: torch.Tensor, text_emb: torch.Tensor,
                            last_token: torch.Tensor) -> torch.Tensor:
        depth, audio, p = self.model.depth, self.audio_tokens, self.params
        slots = depth.expand(hidden, text_emb, last_token)  # (1, G, D)
        tokens = []
        base = self.depth_frame * self.G
        for g in range(self.G):
            _, logits = self.dec.depth.step(depth.input_matching(slots[:, g]), base + g)
            logits = logits.clone()
            if p.max_silent_run >= 0 and self.silent_run >= p.max_silent_run:
                logits[:, self.quiet_mask] = float("-inf")
            tok = sample_top_k(logits, p.audio_temperature, p.audio_top_k)
            self.silent_run = self.silent_run + 1 if int(tok.item()) in audio.quiet else 0
            tokens.append(tok)
            if g < self.G - 1:
                slots[:, g + 1] += depth.embed(tok)
        return torch.stack(tokens, dim=-1)
