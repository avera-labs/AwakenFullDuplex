"""One live conversation: microphone chunks in, model frames and speech out.

Two threads per connection:

* the *loop* thread feeds mic audio to the streaming ear and steps the duplex session once per
  completed 200 ms frame;
* the *speech* thread turns every generated frame into exactly one 200 ms PCM frame (decoded
  speech or silence) through a streaming detokenizer session, so the client plays a constant
  5 Hz stream on one clock. Decoded audio lags its tokens by about two frames (the codec's
  streaming hold-backs); the tail of an utterance drains into the following silence frames.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from awaken_chat.codec.detokenizer import VoicePrompt
from awaken_chat.frame import AUDIO, FRAME_HZ, SAMPLES_PER_FRAME_IN, system_prompt
from awaken_chat.inference.params import DecodeParams
from awaken_chat.inference.session import Decoders, DuplexSession, FrameOut
from awaken_chat.models.ear import StreamingEar
from awaken_chat.models.loaders import LoadedModel
from awaken_chat.server.protocol import audio_class, encode_audio_frame, text_class, text_message

logger = logging.getLogger(__name__)

Emit = Callable[[str | bytes], None]
END_SILENCE_FRAMES = 2 * FRAME_HZ  # an utterance's text is closed after 2 s without words
FRAME_BATCH = 5
STATS_EVERY_S = 1.0


@dataclass(frozen=True)
class DetokSettings:
    freeze_holdback_tok: int = 0  # 0 = lowest latency; 5 = seam-free, +200 ms
    ctx_cap_tok: int | None = 50  # bound the live flow context (fold older tokens)
    fold_keep_tok: int = 25
    run_cap_tok: int = 300  # restart the stream past this (12 s of speech)


class SpeechThread(threading.Thread):
    """Frame-synchronous detokenization: one PCM frame out per generated frame in."""

    def __init__(self, detok, emit: Emit, voice: VoicePrompt | None, settings: DetokSettings) -> None:
        super().__init__(daemon=True, name="awaken-speech")
        self.detok, self.emit, self.voice, self.settings = detok, emit, voice, settings
        self.q: queue.Queue = queue.Queue()
        self.decode_ms: list[float] = []

    def push(self, frame: FrameOut) -> None:
        self.q.put(frame)

    def close(self) -> None:
        self.q.put(None)

    def _new_session(self):
        s = self.settings
        return self.detok.stream_session(self.voice, freeze_holdback_tok=s.freeze_holdback_tok,
                                         ctx_cap_tok=s.ctx_cap_tok, fold_keep_tok=s.fold_keep_tok)

    def run(self) -> None:
        fifo = np.zeros(0, np.float32)
        sess = None
        while True:
            frame = self.q.get()
            if frame is None:
                return
            t0 = time.perf_counter()
            voiced = all(AUDIO.is_code(t) for t in frame.audio_ids)
            if voiced:
                if sess is None:
                    sess = self._new_session()
                new = sess.push(frame.audio_ids).numpy()
                if len(new):
                    fifo = np.concatenate([fifo, new])
                if len(sess.tokens) >= self.settings.run_cap_tok:
                    tail = sess.flush().numpy()
                    if len(tail):
                        fifo = np.concatenate([fifo, tail])
                    sess = self._new_session()
            elif sess is not None:
                tail = sess.flush().numpy()
                if len(tail):
                    fifo = np.concatenate([fifo, tail])
                sess = None
            self.decode_ms.append((time.perf_counter() - t0) * 1000)
            if len(fifo) >= SAMPLES_PER_FRAME_IN:
                pcm, fifo = fifo[:SAMPLES_PER_FRAME_IN], fifo[SAMPLES_PER_FRAME_IN:]
            elif len(fifo) and sess is None:
                pcm = np.concatenate([fifo, np.zeros(SAMPLES_PER_FRAME_IN - len(fifo), np.float32)])
                fifo = fifo[:0]
            else:
                pcm = None  # run onset: samples not yet available
            self.emit(encode_audio_frame(frame.frame, pcm))


class LiveSession:
    def __init__(self, loaded: LoadedModel, decoders: Decoders, detok, emit: Emit,
                 params: DecodeParams, prompt_extra: str = "", voice: VoicePrompt | None = None,
                 seed: int | None = None, detok_settings: DetokSettings | None = None,
                 max_frames: int = 1500) -> None:
        detok_settings = detok_settings or DetokSettings()
        self.loaded, self.emit, self.params = loaded, emit, params
        self.seed = seed if seed is not None else int(torch.randint(0, 1 << 30, (1,)))
        torch.manual_seed(self.seed)
        self.session = DuplexSession(loaded.model, decoders,
                                     loaded.prompt_ids(system_prompt(prompt_extra)), params=params)
        self.ear = StreamingEar(loaded.model.ear, loaded.feature_extractor)
        self.speech = SpeechThread(detok, emit, voice, detok_settings)
        self.max_frames = max_frames
        self.in_q: queue.Queue = queue.Queue()
        self.stop_flag = False
        self.last_seen = time.time()
        self.n_in_samples = 0
        self.step_ms: list[float] = []
        self.ear_ms: list[float] = []
        self._batch: list[dict] = []
        self._utt_text: list[str] = []
        self._utt_start = 0
        self._utt_last_word = 0
        self._utt_last_audio = 0
        self._last_stats = time.time()

    # Input ----------------------------------------------------------------------------
    def push_audio(self, pcm: np.ndarray) -> None:
        self.last_seen = time.time()
        self.n_in_samples += len(pcm)
        self.in_q.put(pcm)

    def stop(self) -> None:
        self.stop_flag = True

    # Output ---------------------------------------------------------------------------
    def _report_frame(self, out: FrameOut) -> None:
        tc = text_class(out.text_id)
        word = self.loaded.tokenizer.decode([out.text_id]) if tc == "txt" else None
        self._batch.append({"f": out.frame, "t": round(out.frame / FRAME_HZ, 1), "text": word,
                            "text_class": tc, "audio_class": [audio_class(a) for a in out.audio_ids]})
        if len(self._batch) >= FRAME_BATCH:
            self.emit(text_message("frames", items=self._batch))
            self._batch = []
        audio_on = any(AUDIO.is_code(a) or a == AUDIO.bos for a in out.audio_ids)
        words_done = bool(self._utt_text) and out.frame - self._utt_last_word >= END_SILENCE_FRAMES
        if word is not None:
            if words_done:
                self._close_utterance()
            if not self._utt_text:
                self._utt_start = self._utt_last_audio = out.frame
            self._utt_text.append(word)
            self._utt_last_word = out.frame
        elif words_done and not audio_on:
            self._close_utterance()
        if self._utt_text and audio_on:
            self._utt_last_audio = out.frame
        self.speech.push(out)

    def _close_utterance(self) -> None:
        text = "".join(self._utt_text).strip()
        if text:
            end = max(self._utt_last_word, self._utt_last_audio)
            self.emit(text_message("utterance", text=text, t0=round(self._utt_start / FRAME_HZ, 1),
                                   t1=round((end + 1) / FRAME_HZ, 1)))
        self._utt_text = []

    def _stats(self) -> None:
        now = time.time()
        if now - self._last_stats < STATS_EVERY_S:
            return
        self._last_stats = now
        lag = self.n_in_samples // SAMPLES_PER_FRAME_IN - self.session.frame

        def p(xs: list[float]) -> dict:
            if not xs:
                return {}
            v = np.array(xs[-50:])
            return {"p50": round(float(np.median(v))), "p95": round(float(np.percentile(v, 95)))}

        self.emit(text_message("stats", frame=self.session.frame, lag_frames=int(lag),
                               step_ms=p(self.step_ms), ear_ms=p(self.ear_ms),
                               detok_ms=p(self.speech.decode_ms)))

    # The loop -------------------------------------------------------------------------
    def run(self, keepalive_s: float = 75.0) -> None:
        self.speech.start()
        try:
            while not self.stop_flag:
                try:
                    pcm = self.in_q.get(timeout=0.5)
                except queue.Empty:
                    self._stats()
                    if time.time() - self.last_seen > keepalive_s:
                        logger.warning("no client audio or ping for %.0f s, ending session", keepalive_s)
                        break
                    continue
                t0 = time.perf_counter()
                feats = self.ear.feed(pcm)
                self.ear_ms.append((time.perf_counter() - t0) * 1000)
                for feat in feats:
                    if self.stop_flag or self.session.frame >= self.max_frames:
                        self.stop_flag = True
                        break
                    t1 = time.perf_counter()
                    out = self.session.step(feat)
                    self.step_ms.append((time.perf_counter() - t1) * 1000)
                    self._report_frame(out)
                self._stats()
        except Exception as e:  # noqa: BLE001
            logger.exception("live session died")
            self.emit(text_message("error", message=f"{type(e).__name__}: {e}"))
        finally:
            self._close_utterance()
            if self._batch:
                self.emit(text_message("frames", items=self._batch))
            self.speech.close()
            self.speech.join(timeout=5)
            self.emit(text_message("end", frames=self.session.frame))
