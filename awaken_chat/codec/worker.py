"""The detokenizer in its own process.

The live server steps the model in one thread and decodes speech in another. In one process
the two couple through the GIL (the flow is a burst of small kernel launches, so a CPU-busy
thread slows it by tens of milliseconds) and through torch.compile's process-wide tracing lock.
A separate process removes both: the pipe round-trip is well under a millisecond.

Protocol (multiprocessing Pipe, spawn): ``(op, *args) -> result``. Exceptions are returned as
values and re-raised in the parent, so one failed request never wedges the worker.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time

import numpy as np
import torch

from awaken_chat.codec.detokenizer import CODEC_REPO, VoicePrompt
from awaken_chat.frame import AUDIO, GROUP_SIZE

logger = logging.getLogger(__name__)


def _worker(conn, repo_or_dir: str, fp16: bool, warm_frames: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s detok %(levelname)s %(message)s")
    from awaken_chat.codec.detokenizer import AudioDetokenizer

    d = AudioDetokenizer.load(repo_or_dir, fp16=fp16)
    sessions = {}
    while True:
        try:
            op, *args = conn.recv()
        except EOFError:
            return  # parent gone
        try:
            if op == "warm":
                t0 = time.time()
                # Mirror live use: streaming pushes of one frame, capped and uncapped windows.
                for cap, keep in ((None, 50), (50, 25), (100, 50)):
                    s = d.stream_session(None, freeze_holdback_tok=0, ctx_cap_tok=cap, fold_keep_tok=keep)
                    for _ in range(warm_frames):
                        s.push(torch.randint(0, AUDIO.codebook_size, (GROUP_SIZE,)).tolist())
                    s.flush()
                torch.cuda.synchronize()
                conn.send(time.time() - t0)
            elif op == "open":
                sid, voice, kwargs = args
                sessions[sid] = d.stream_session(voice, **kwargs)
                conn.send(True)
            elif op == "push":
                sid, toks = args
                conn.send(sessions[sid].push(toks).numpy())
            elif op == "flush":
                s = sessions.pop(args[0], None)
                conn.send(s.flush().numpy() if s is not None else np.zeros(0, np.float32))
            elif op == "close":
                sessions.pop(args[0], None)
                conn.send(True)
            elif op == "stop":
                conn.send(True)
                return
            else:
                conn.send(RuntimeError(f"unknown op {op}"))
        except Exception as e:  # noqa: BLE001 - report to the parent, keep serving
            logger.exception("detok op %s failed", op)
            conn.send(e)


class DetokProcess:
    """Parent-side handle. Calls are serialized by a lock (one request in flight)."""

    def __init__(self, repo_or_dir: str = CODEC_REPO, fp16: bool = True, warm_frames: int = 50) -> None:
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child, repo_or_dir, fp16, warm_frames),
                                daemon=True, name="awaken-detok")
        self.proc.start()
        child.close()
        self._next_sid = 0
        self._lock = threading.Lock()
        logger.info("detokenizer process started (pid %d)", self.proc.pid)

    def _call(self, op, *args):
        with self._lock:
            self.conn.send((op, *args))
            result = self.conn.recv()
        if isinstance(result, Exception):
            raise result
        return result

    def warm(self) -> float:
        """Load + warm every live shape; returns the seconds it took."""
        return self._call("warm")

    def stream_session(self, voice: VoicePrompt | None = None, **kwargs) -> ProxySession:
        self._next_sid += 1
        return ProxySession(self, self._next_sid, voice, kwargs)

    def alive(self) -> bool:
        return self.proc.is_alive()

    def stop(self) -> None:
        try:
            if self.proc.is_alive():
                self.conn.send(("stop",))
                self.proc.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        if self.proc.is_alive():
            self.proc.kill()
        logger.info("detokenizer process stopped")


class ProxySession:
    """Same surface as :class:`~awaken_chat.codec.streaming.StreamingDetokSession`."""

    def __init__(self, proc: DetokProcess, sid: int, voice: VoicePrompt | None, kwargs: dict) -> None:
        self.proc, self.sid = proc, sid
        self.tokens: list[int] = []
        proc._call("open", sid, voice, kwargs)

    def push(self, tokens) -> torch.Tensor:
        tokens = [int(t) for t in tokens]
        self.tokens.extend(tokens)
        return torch.from_numpy(self.proc._call("push", self.sid, tokens))

    def flush(self) -> torch.Tensor:
        return torch.from_numpy(self.proc._call("flush", self.sid))
