"""The live server: a FastAPI app with one WebSocket conversation at a time.

The model, its static decoders and the detokenizer process are loaded lazily on the first
conversation and kept for the life of the process. One conversation runs at a time: the CUDA
graphs and static caches belong to a single session.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.websockets import WebSocketDisconnect

from awaken_chat.codec.detokenizer import CODEC_REPO, VoicePrompt
from awaken_chat.inference.params import PRESETS, DecodeParams, SessionLimits
from awaken_chat.inference.session import Decoders
from awaken_chat.models.loaders import LoadedModel, load_model
from awaken_chat.server.protocol import StartRequest, decode_mic, params_dict, text_message
from awaken_chat.server.session_runner import DetokSettings, LiveSession

logger = logging.getLogger(__name__)
ASSET_VOICES = Path(__file__).resolve().parents[2] / "assets" / "voices"


class ServerSettings(BaseSettings):
    """Configuration, overridable from the environment with the ``AWAKEN_`` prefix."""

    model_config = SettingsConfigDict(env_prefix="AWAKEN_")

    model: str  # local checkpoint directory or Hub repo id
    codec: str = CODEC_REPO
    host: str = "0.0.0.0"
    port: int = 8998
    ssl_dir: str = ""  # directory holding key.pem + cert.pem (browsers need https for the mic)
    voices_dir: str = str(ASSET_VOICES)  # *.pt voice prompts
    default_voice: str = ""
    detok_fp16: bool = True
    backbone_len: int = SessionLimits().backbone_len
    depth_len: int = SessionLimits().depth_len
    max_session_s: float = 300.0


class Runtime:
    """Everything loaded once per process."""

    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self.loaded: LoadedModel | None = None
        self.decoders: Decoders | None = None
        self.detok = None
        self.voices: dict[str, VoicePrompt] = {}
        self.load_lock = threading.Lock()
        self.live_lock = threading.Lock()
        self.current: LiveSession | None = None

    @property
    def ready(self) -> bool:
        return self.loaded is not None

    def list_voices(self) -> list[str]:
        d = Path(self.settings.voices_dir)
        return sorted(p.stem for p in d.glob("*.pt")) if d.is_dir() else []

    def voice(self, name: str) -> VoicePrompt | None:
        name = name or self.settings.default_voice
        if not name:
            return None
        if name not in self.voices:
            path = Path(self.settings.voices_dir) / f"{name}.pt"
            if not path.is_file():
                raise ValueError(f"unknown voice {name!r}")
            self.voices[name] = VoicePrompt.load(path)
        return self.voices[name]

    def ensure_loaded(self, status) -> None:
        with self.load_lock:
            if self.ready:
                return
            from awaken_chat.codec.worker import DetokProcess

            status("starting the detokenizer process")
            detok = DetokProcess(self.settings.codec, fp16=self.settings.detok_fp16)
            status("loading the model")
            loaded = load_model(self.settings.model)
            status("capturing decode graphs")
            limits = SessionLimits(self.settings.backbone_len, self.settings.depth_len)
            decoders = Decoders(loaded.model, limits)
            self._warmup(loaded, decoders)
            status("warming up the detokenizer")
            logger.info("detokenizer warm in %.0f s", detok.warm())
            self.loaded, self.decoders, self.detok = loaded, decoders, detok
            free, total = torch.cuda.mem_get_info()
            logger.info("ready; GPU %.1f / %.1f GiB used", (total - free) / 2**30, total / 2**30)

    @staticmethod
    def _warmup(loaded: LoadedModel, decoders: Decoders) -> None:
        """Capture both CUDA graphs on an idle and a voiced frame."""
        from awaken_chat.frame import SYSTEM_PROMPT
        from awaken_chat.inference.session import DuplexSession

        t0 = time.time()
        sess = DuplexSession(loaded.model, decoders, loaded.prompt_ids(SYSTEM_PROMPT))
        feat = torch.zeros(loaded.model.group_size, loaded.model.config.hidden_size,
                           device=loaded.device, dtype=loaded.model.dtype)
        for i in range(12):
            sess.next_text = torch.tensor([1000 if i >= 6 else sess.text_tokens.pad], device=loaded.device)
            sess.step(feat)
        torch.cuda.synchronize()
        logger.info("decode graphs captured in %.0f s", time.time() - t0)


def create_app(settings: ServerSettings) -> FastAPI:
    app = FastAPI(title="Awaken Chat")
    rt = Runtime(settings)
    app.state.runtime = rt

    @app.get("/health")
    async def health():
        free, total = torch.cuda.mem_get_info()
        return JSONResponse({"model": settings.model, "ready": rt.ready, "live": rt.live_lock.locked(),
                             "voices": rt.list_voices(), "gpu_used_gib": round((total - free) / 2**30, 1)})

    @app.get("/voices")
    async def voices():
        return JSONResponse({"voices": rt.list_voices(), "default": settings.default_voice})

    @app.websocket("/ws")
    async def ws_conversation(ws: WebSocket):
        await ws.accept()
        loop = asyncio.get_running_loop()
        out_q: asyncio.Queue = asyncio.Queue()
        sess: LiveSession | None = None
        thread: threading.Thread | None = None
        disconnected = False

        def emit(msg: str | bytes) -> None:
            loop.call_soon_threadsafe(out_q.put_nowait, msg)

        async def sender() -> None:
            while True:
                msg = await out_q.get()
                try:
                    if isinstance(msg, bytes):
                        await ws.send_bytes(msg)
                    else:
                        await ws.send_text(msg)
                finally:
                    out_q.task_done()

        send_task = asyncio.create_task(sender())
        try:
            while True:
                packet = await ws.receive()
                if packet.get("type") == "websocket.disconnect":
                    disconnected = True
                    break
                if packet.get("bytes") is not None:
                    if sess is not None:
                        sess.push_audio(decode_mic(packet["bytes"]))
                    continue
                import json

                msg = json.loads(packet.get("text") or "{}")
                kind = msg.get("type")
                if kind == "start":
                    if sess is not None:
                        emit(text_message("error", message="already started"))
                        continue
                    req = StartRequest.parse(msg)
                    try:
                        await asyncio.to_thread(rt.ensure_loaded,
                                                lambda m: emit(text_message("status", message=m)))
                        base = PRESETS.get(req.preset, PRESETS["live"])
                        params = base.with_overrides(**_clean_overrides(req.decode, base))
                        voice = rt.voice(req.voice)
                    except Exception as e:  # noqa: BLE001
                        logger.exception("start failed")
                        emit(text_message("error", message=f"start failed: {e}"))
                        continue
                    if rt.live_lock.locked():
                        emit(text_message("error", message="another conversation is running"))
                        continue
                    rt.live_lock.acquire()
                    sess = LiveSession(rt.loaded, rt.decoders, rt.detok, emit, params, req.prompt,
                                       voice, req.seed, DetokSettings(),
                                       max_frames=int(settings.max_session_s * 5))
                    rt.current = sess
                    emit(text_message("ready", seed=sess.seed, decode=params_dict(params)))
                    thread = threading.Thread(target=_run_and_release, args=(sess, rt), daemon=True)
                    thread.start()
                elif kind == "ping":
                    if sess is not None:
                        sess.last_seen = time.time()
                elif kind == "stop":
                    break
        except WebSocketDisconnect:
            disconnected = True
        finally:
            if sess is not None:
                sess.stop()
            if thread is not None:
                # Keep the event loop sending while the worker emits its final events.
                await asyncio.to_thread(thread.join, timeout=5)
            await asyncio.sleep(0)  # deliver the worker's call_soon_threadsafe callbacks
            if not disconnected and not send_task.done():
                try:
                    await asyncio.wait_for(out_q.join(), timeout=2)
                except asyncio.TimeoutError:
                    logger.warning("timed out sending final WebSocket events")
            send_task.cancel()
            await asyncio.gather(send_task, return_exceptions=True)
            logger.info("connection closed")

    return app


def _run_and_release(sess: LiveSession, rt: Runtime) -> None:
    try:
        sess.run()
    finally:
        rt.current = None
        rt.live_lock.release()


def _clean_overrides(raw: dict, base: DecodeParams) -> dict:
    """Keep only known DecodeParams fields, cast to the field's type."""
    out = {}
    for key, value in raw.items():
        if not hasattr(base, key):
            continue
        kind = type(getattr(base, key))
        try:
            out[key] = kind(value)
        except (TypeError, ValueError):
            continue
    return out
