"""Serve the playground and proxy the existing inference protocol, without importing the model."""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import math
import ssl
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web

STATIC = Path(__file__).parent / "static"
CLIENT = web.AppKey("client", ClientSession)
UPSTREAM = web.AppKey("upstream", str)
PUBLIC_ORIGIN = web.AppKey("public_origin", str)
SOCKETS = web.AppKey("sockets", set)
DIAGNOSTICS = web.AppKey("diagnostics", dict)
ASSETS = {"app.js", "audio.js", "capture-worklet.js", "core.js", "style.css"}


def validate_upstream(value: str) -> str:
    url = urlsplit(value)
    if url.scheme not in {"ws", "wss"} or not url.hostname or url.username or url.password:
        raise ValueError("upstream must be a ws:// or wss:// URL without credentials")
    if url.query or url.fragment:
        raise ValueError("upstream must not have a query string or fragment")
    _ = url.port  # Validate malformed ports at startup.
    return value


def api_url(upstream: str, endpoint: str) -> str:
    url = urlsplit(upstream)
    parent = url.path.rstrip("/").rsplit("/", 1)[0]
    return urlunsplit(
        ("https" if url.scheme == "wss" else "http", url.netloc, f"{parent}/{endpoint}", "", "")
    )


def validate_public_origin(value: str) -> str:
    if not value:
        return ""
    url = urlsplit(value)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        raise ValueError("public origin must be an http:// or https:// origin without credentials or a path")
    _ = url.port
    return urlunsplit((url.scheme, url.netloc, "", "", ""))


@web.middleware
async def response_headers(request: web.Request, handler):
    response = await handler(request)
    if not response.prepared:
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
    return response


async def config(request: web.Request) -> web.Response:
    return web.json_response(
        {"upstream": request.app[UPSTREAM], "sample_rate": 24000, "frame_samples": 4800, "websocket": "/ws"}
    )


async def metadata(request: web.Request) -> web.Response:
    endpoint = request.match_info["endpoint"]
    try:
        async with request.app[CLIENT].get(
            api_url(request.app[UPSTREAM], endpoint), timeout=ClientTimeout(total=10)
        ) as response:
            if response.status != 200:
                return web.json_response(
                    {"error": f"Inference /{endpoint} returned HTTP {response.status}"}, status=502
                )
            return web.json_response(await response.json())
    except (ClientError, asyncio.TimeoutError, ValueError):
        return web.json_response(
            {"error": f"Cannot read inference /{endpoint}; check --upstream"}, status=502
        )


def observe_packet(state: dict, direction: str, payload: str | bytes) -> None:
    """Keep signal counters for debugging, never retain PCM or conversation text."""
    if isinstance(payload, bytes):
        if direction == "input" and len(payload) > 1 and payload[0] == 1 and (len(payload) - 1) % 2 == 0:
            pcm = array.array("h", payload[1:])
            if sys.byteorder != "little":
                pcm.byteswap()
            level = math.sqrt(sum(x * x for x in pcm) / len(pcm)) / 32768
            peak = max(abs(x) for x in pcm) / 32768
            state["input_samples"] += len(pcm)
            state["input_rms"] = round(level, 6)
            state["input_peak"] = round(peak, 6)
            state["input_peak_max"] = max(state["input_peak_max"], round(peak, 6))
            state["last_input_at"] = time.time()
            if level >= 0.001:
                state["signal_samples"] += len(pcm)
        elif direction == "output" and len(payload) >= 5 and payload[0] in {2, 3}:
            state["output_frames"] += 1
            if payload[0] == 2:
                state["voiced_frames"] += 1
        return
    try:
        message = json.loads(payload)
    except (ValueError, TypeError):
        return
    if not isinstance(message, dict):
        return
    kind = message.get("type")
    if direction == "output":
        if kind == "ready":
            state["ready"] = True
        elif kind == "frames":
            items = message.get("items")
            if isinstance(items, list):
                state["text_tokens"] += sum(
                    item.get("text_class") == "txt" for item in items if isinstance(item, dict)
                )
        elif kind == "stats":
            state["model_frames"] = message.get("frame")
        elif kind == "error":
            state["error"] = str(message.get("message", ""))[:500]


async def diagnostics(request: web.Request) -> web.Response:
    return web.json_response(request.app[DIAGNOSTICS].get("last", {}))


async def relay(source, destination, state: dict, direction: str) -> None:
    async for message in source:
        if message.type == WSMsgType.TEXT:
            await destination.send_str(message.data)
            observe_packet(state, direction, message.data)
        elif message.type == WSMsgType.BINARY:
            await destination.send_bytes(message.data)
            observe_packet(state, direction, message.data)
        elif message.type == WSMsgType.ERROR:
            raise ConnectionError("WebSocket transport failed")


async def websocket(request: web.Request) -> web.WebSocketResponse:
    # A developer service should not let an unrelated website open the local model's socket.
    origin = request.headers.get("Origin")
    allowed_origins = {f"{request.scheme}://{request.host}"}
    if request.app[PUBLIC_ORIGIN]:
        allowed_origins.add(request.app[PUBLIC_ORIGIN])
    if origin and origin not in allowed_origins:
        raise web.HTTPForbidden(text="WebSocket origin must match this playground")
    downstream = web.WebSocketResponse(heartbeat=25, max_msg_size=1024 * 1024)
    await downstream.prepare(request)
    request.app[SOCKETS].add(downstream)
    state = {
        "connected": True, "started_at": time.time(), "ready": False,
        "input_samples": 0, "signal_samples": 0, "input_rms": 0.0, "input_peak": 0.0,
        "input_peak_max": 0.0, "output_frames": 0, "voiced_frames": 0, "text_tokens": 0,
    }
    request.app[DIAGNOSTICS]["last"] = state
    tasks = []
    try:
        async with request.app[CLIENT].ws_connect(
            request.app[UPSTREAM], heartbeat=25, max_msg_size=1024 * 1024
        ) as upstream:
            tasks = [
                asyncio.create_task(relay(downstream, upstream, state, "input")),
                asyncio.create_task(relay(upstream, downstream, state, "output")),
            ]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            # Closing the client also closes its upstream and releases the inference session.
    except (ClientError, asyncio.TimeoutError, ConnectionError):
        if not downstream.closed:
            await downstream.send_json(
                {
                    "type": "error",
                    "message": "Inference connection failed; check --upstream and the inference server logs",
                }
            )
            await downstream.close(code=1011, message=b"Inference connection failed")
    finally:
        state["connected"] = False
        state["ended_at"] = time.time()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        request.app[SOCKETS].discard(downstream)
        await downstream.close()
    return downstream


async def lifecycle(app: web.Application):
    async with ClientSession(timeout=ClientTimeout(total=None, sock_connect=10)) as client:
        app[CLIENT] = client
        yield


async def shutdown(app: web.Application) -> None:
    await asyncio.gather(
        *(socket.close(code=1001, message=b"Playground shutting down") for socket in list(app[SOCKETS]))
    )


async def static_file(request: web.Request) -> web.FileResponse:
    name = request.match_info.get("name", "index.html")
    if name != "index.html" and name not in ASSETS:
        raise web.HTTPNotFound()
    return web.FileResponse(STATIC / name)


def create_app(upstream: str = "ws://127.0.0.1:8998/ws", public_origin: str = "") -> web.Application:
    app = web.Application(middlewares=[response_headers])
    app[UPSTREAM] = validate_upstream(upstream)
    app[PUBLIC_ORIGIN] = validate_public_origin(public_origin)
    app[SOCKETS] = set()
    app[DIAGNOSTICS] = {}
    app.cleanup_ctx.append(lifecycle)
    app.on_shutdown.append(shutdown)
    app.router.add_get("/api/config", config)
    app.router.add_get("/api/diagnostics", diagnostics)
    app.router.add_get("/api/{endpoint:health|voices}", metadata)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/", static_file)
    app.router.add_get("/{name}", static_file)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="ws://127.0.0.1:8998/ws", help="Inference WebSocket URL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--public-origin", default="",
        help="Exact browser origin to allow behind a TLS-terminating proxy, e.g. https://pod.proxy.runpod.net",
    )
    parser.add_argument("--ssl-cert", type=Path, help="HTTPS certificate (optional on localhost)")
    parser.add_argument("--ssl-key", type=Path)
    args = parser.parse_args()
    if bool(args.ssl_cert) != bool(args.ssl_key):
        parser.error("--ssl-cert and --ssl-key must be provided together")
    try:
        app = create_app(args.upstream, args.public_origin)
    except ValueError as error:
        parser.error(str(error))
    context = None
    if args.ssl_cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.ssl_cert, args.ssl_key)
    web.run_app(app, host=args.host, port=args.port, ssl_context=context)


if __name__ == "__main__":
    main()
