"""Incremental token -> waveform decoding for one voiced run.

CosyVoice3's flow is chunk-causal: mel for the newest tokens is still being revised while the
next tokens arrive (the pre-lookahead layer holds back the last 3 tokens, and the DiT attends in
50-frame chunks). Decoding each 200 ms frame independently would emit the flow's lookahead-less
tail every time, one audible seam per frame. This session instead:

* runs the flow (``finalize=False``) over the whole accumulated run every push,
* freezes mel only up to ``produced - 2 * freeze_holdback_tok`` frames (newer frames may still
  change),
* re-runs the causal HiFT vocoder on the frozen mel and emits samples monotonically from a
  speech offset, so the output is prefix-stable with no cross-fades.

Emitted audio therefore lags the token frontier by a few tokens. ``flush`` finalizes both
stages and returns the tail.

Because every push re-decodes the whole run, cost grows with run length. ``ctx_cap_tok`` bounds
it: once the live window exceeds the cap, the oldest tokens are folded into the flow *prompt*
(the zero-shot conditioning path the flow was trained with) so decoding continues conditioned on
its own recent output rather than restarting cold.
"""

from __future__ import annotations

import torch

from awaken_chat.codec.detokenizer import MEL_DIM, AudioDetokenizer, VoicePrompt


class StreamingDetokSession:
    MIN_MEL = 12  # HiFT's causal convs need this much context to start

    def __init__(self, detokenizer: AudioDetokenizer, voice: VoicePrompt | None = None,
                 freeze_holdback_tok: int = 0, ctx_cap_tok: int | None = 50,
                 fold_keep_tok: int = 25) -> None:
        self.d = detokenizer
        self.holdback = freeze_holdback_tok
        self.cap = ctx_cap_tok
        self.keep = fold_keep_tok
        dev = detokenizer.device
        self.voice = voice.to(dev) if voice is not None else VoicePrompt.empty(dev)
        self.prompt_token, self.prompt_feat = self.voice.prompt_token, self.voice.prompt_feat
        self.tokens: list[int] = []
        self.origin = 0  # tokens folded out of the live window
        self.mel_buf: torch.Tensor | None = None  # frozen mel, run-absolute
        self.mel_seen = 0  # run-absolute frozen mel frames
        self.speech_off = 0
        self.done = False

    def _maybe_fold(self) -> None:
        if self.cap is None or len(self.tokens) - self.origin <= self.cap:
            return
        new_origin = min(len(self.tokens) - self.keep, self.mel_seen // 2)
        if new_origin <= self.origin:
            return
        self.origin = new_origin
        lo = max(0, self.origin - self.keep)
        dev = self.d.device
        fold_tok = torch.tensor(self.tokens[lo:self.origin], dtype=torch.int32, device=dev)[None]
        # Tokens-only fold: conditioning on our own frozen mel creeps louder at every fold
        # (positive feedback); token context alone keeps loudness identical to an unbounded
        # decode, so the folded mel is zeros.
        fold_mel = torch.zeros(1, 2 * (self.origin - lo), MEL_DIM, device=dev)
        self.prompt_token = torch.cat([self.voice.prompt_token, fold_tok], dim=1)
        self.prompt_feat = torch.cat(
            [self.voice.prompt_feat, fold_mel.to(self.voice.prompt_feat.dtype)], dim=1)

    def _step(self, finalize: bool) -> torch.Tensor:
        empty = torch.zeros(0, dtype=torch.float32)
        if len(self.tokens) <= self.origin:
            return empty
        dev = self.d.device
        toks = torch.tensor(self.tokens[self.origin:], dtype=torch.int32, device=dev)[None]
        voice = VoicePrompt(self.prompt_token, self.prompt_feat, self.voice.embedding)
        mel = self.d._flow(toks, voice, finalize=finalize)
        # mel covers the live tokens only; index run-absolute via origin
        limit = mel.shape[2] if finalize else max(0, mel.shape[2] - 2 * self.holdback)
        new = mel[:, :, self.mel_seen - 2 * self.origin:limit]
        if new.shape[2]:
            self.mel_buf = new if self.mel_buf is None else torch.cat([self.mel_buf, new], dim=2)
            self.mel_seen = 2 * self.origin + limit
        if self.mel_buf is None or (not finalize and self.mel_buf.shape[2] < self.MIN_MEL):
            return empty
        speech = self.d._vocode(self.mel_buf, finalize=finalize)
        out = speech[:, self.speech_off:].squeeze(0).float().cpu()
        self.speech_off += out.shape[0]
        return out

    def push(self, tokens) -> torch.Tensor:
        """Feed newly generated codec tokens; returns the newly available samples (may be empty)."""
        if self.done:
            raise RuntimeError("session already flushed")
        self.tokens.extend(int(t) for t in tokens)
        out = self._step(finalize=False)
        self._maybe_fold()
        return out

    def flush(self) -> torch.Tensor:
        """Finalize the run: decode the held-back mel and HiFT's tail."""
        if self.done:
            return torch.zeros(0, dtype=torch.float32)
        self.done = True
        return self._step(finalize=True)
