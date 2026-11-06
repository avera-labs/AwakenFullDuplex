"""Protocol simulator for local tests: emits a sine tone, NOT model-generated speech.

Run with: uv run --project web_client python web_client/tests/mock_inference.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct

from aiohttp import WSMsgType, web

STATE = web.AppKey("state", dict)
FRAMES = web.AppKey("frames", list)
MESSAGES = web.AppKey("messages", list)
SAMPLE_RATE = 24000
FRAME_SAMPLES = 4800


def create_mock(
    *, busy: bool = False, max_frames: int = 1500, fail_after: int | None = None, ready_delay: float = 0.01
) -> web.Application:
    app = web.Application()
    app[STATE] = {"live": busy, "fail_after": fail_after, "ready_delay": ready_delay,
                  "offline": False, "holdback_samples": 0}
    app[FRAMES] = []
    app[MESSAGES] = []

    async def health(_):
        if app[STATE]["offline"]:
            raise web.HTTPServiceUnavailable()
        return web.json_response(
            {
                "model": "PROTOCOL SIMULATOR (no model)",
                "ready": True,
                "live": app[STATE]["live"],
                "voices": ["test-voice"],
                "gpu_used_gib": 0,
            }
        )

    async def voices(_):
        return web.json_response({"voices": ["test-voice"], "default": ""})

    async def conversation(request):
        if app[STATE]["offline"]:
            raise web.HTTPServiceUnavailable()
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        frame = 0
        samples = 0
        started = False
        batch = []

        async def end():
            if batch:
                await ws.send_json({"type": "frames", "items": list(batch)})
                batch.clear()
            if frame > 2:
                await ws.send_json(
                    {
                        "type": "utterance",
                        "text": "This is a test reply from the protocol simulator.",
                        "t0": 0.4,
                        "t1": round(frame / 5, 1),
                    }
                )
            await ws.send_json({"type": "end", "frames": frame})
            await ws.close()

        try:
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    value = json.loads(message.data)
                    app[MESSAGES].append(value)
                    kind = value.get("type")
                    if kind == "start":
                        if app[STATE]["live"]:
                            await ws.send_json(
                                {"type": "error", "message": "another conversation is running"}
                            )
                            continue
                        app[STATE]["live"] = True
                        started = True
                        await ws.send_json(
                            {"type": "status", "message": "Protocol simulator · No model loaded"}
                        )
                        await asyncio.sleep(app[STATE]["ready_delay"])
                        await ws.send_json(
                            {
                                "type": "ready",
                                "seed": value.get("seed", 42),
                                "decode": {"text_temperature": 0.5, **value.get("decode", {})},
                            }
                        )
                    elif kind == "stop":
                        await end()
                        break
                elif message.type == WSMsgType.BINARY and started:
                    payload = message.data
                    if not payload or payload[0] != 1 or (len(payload) - 1) % 2:
                        await ws.send_json({"type": "error", "message": "invalid mic PCM"})
                        continue
                    samples += (len(payload) - 1) // 2
                    # StreamingEar needs future input before it can finalize the current frame.
                    while samples >= FRAME_SAMPLES + app[STATE]["holdback_samples"]:
                        samples -= FRAME_SAMPLES
                        app[FRAMES].append(frame)
                        failure_frame = app[STATE]["fail_after"]
                        if failure_frame is not None and frame >= failure_frame:
                            await ws.send_json({"type": "error", "message": "simulated inference failure"})
                            await ws.close(code=1011)
                            return ws
                        silent = frame < 2 or frame % 10 > 6
                        audio = (
                            b""
                            if silent
                            else b"".join(
                                struct.pack("<h", round(4000 * math.sin(2 * math.pi * 240 * i / SAMPLE_RATE)))
                                for i in range(FRAME_SAMPLES)
                            )
                        )
                        await ws.send_bytes(struct.pack("<BI", 3 if silent else 2, frame) + audio)
                        batch.append(
                            {
                                "f": frame,
                                "t": round(frame / 5, 1),
                                "text": (
                                    "This is a test reply from the protocol simulator."
                                    if frame == 2 else None
                                ),
                                "text_class": "txt" if frame == 2 else "pad",
                                "audio_class": ["pad" if silent else "voi"] * 5,
                            }
                        )
                        frame += 1
                        if frame % 5 == 0:
                            await ws.send_json({"type": "frames", "items": list(batch)})
                            batch.clear()
                            await ws.send_json(
                                {
                                    "type": "stats",
                                    "frame": frame,
                                    "lag_frames": 0,
                                    "step_ms": {"p50": 12, "p95": 18},
                                    "ear_ms": {"p50": 5, "p95": 8},
                                    "detok_ms": {"p50": 22, "p95": 35},
                                }
                            )
                        if frame >= max_frames:
                            await end()
                            return ws
        finally:
            if started:
                app[STATE]["live"] = False
        return ws

    app.router.add_get("/health", health)
    app.router.add_get("/voices", voices)
    app.router.add_get("/ws", conversation)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8999)
    parser.add_argument("--busy", action="store_true")
    parser.add_argument("--max-frames", type=int, default=1500)
    parser.add_argument("--fail-after", type=int)
    args = parser.parse_args()
    web.run_app(
        create_mock(busy=args.busy, max_frames=args.max_frames, fail_after=args.fail_after),
        host="127.0.0.1",
        port=args.port,
    )
