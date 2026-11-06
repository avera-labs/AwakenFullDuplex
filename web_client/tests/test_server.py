"""Real HTTP/WebSocket integration, using the inference protocol simulator; no GPU imports."""

from __future__ import annotations

import asyncio
import json
import struct
import sys
import unittest
from pathlib import Path

from aiohttp import WSMsgType, WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import api_url, create_app, validate_public_origin, validate_upstream
from tests.mock_inference import FRAMES, MESSAGES, STATE, create_mock


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.upstream = TestServer(create_mock())
        await self.upstream.start_server()
        self.client = TestClient(
            TestServer(create_app(str(self.upstream.make_url("/ws")).replace("http:", "ws:")))
        )
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.upstream.close()

    async def receive_json(self, socket, kind):
        while not socket.closed:
            message = await socket.receive(timeout=3)
            if message.type == WSMsgType.TEXT:
                value = json.loads(message.data)
                if value.get("type") == kind:
                    return value
            elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED}:
                break
        self.fail(f"Missing {kind}")

    async def test_static_and_metadata(self):
        for path in ["/", "/app.js", "/audio.js", "/core.js", "/capture-worklet.js", "/style.css"]:
            response = await self.client.get(path)
            self.assertEqual(response.status, 200, path)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        config = await (await self.client.get("/api/config")).json()
        self.assertEqual(config["sample_rate"], 24000)
        health = await (await self.client.get("/api/health")).json()
        self.assertIn("SIMULATOR", health["model"])
        voices = await (await self.client.get("/api/voices")).json()
        self.assertEqual(voices["voices"], ["test-voice"])
        for path in ["/server.py", "/pyproject.toml", "/tests/mock_inference.py"]:
            self.assertEqual((await self.client.get(path)).status, 404)

    async def test_full_duplex_frames_and_stop(self):
        socket = await self.client.ws_connect("/ws")
        await socket.send_json({"type": "start", "seed": 73, "decode": {"text_temperature": 0.2}})
        ready = await self.receive_json(socket, "ready")
        self.assertEqual(ready["seed"], 73)
        self.assertEqual(ready["decode"]["text_temperature"], 0.2)

        # Send input while the independent receiver is consuming output, as the UI does.
        async def producer():
            for _ in range(12):
                await socket.send_bytes(b"\x01" + struct.pack("<2400h", *([1000] * 2400)))
                await asyncio.sleep(0.005)
            await socket.send_json({"type": "ping"})
            await socket.send_json({"type": "stop"})

        producer_task = asyncio.create_task(producer())
        audio, text, ended = [], [], False

        async def collect():
            nonlocal ended
            async for message in socket:
                if message.type == WSMsgType.BINARY:
                    kind, frame = struct.unpack_from("<BI", message.data)
                    self.assertEqual(len(message.data), 5 if kind == 3 else 9605)
                    audio.append((kind, frame))
                elif message.type == WSMsgType.TEXT:
                    value = json.loads(message.data)
                    text.append(value["type"])
                    if value["type"] == "end":
                        self.assertEqual(value["frames"], 6)
                        ended = True

        await asyncio.wait_for(collect(), timeout=5)
        await producer_task
        self.assertEqual([frame for _, frame in audio], list(range(6)))
        self.assertIn((3, 0), audio)
        self.assertIn((2, 2), audio)
        self.assertIn("frames", text)
        self.assertIn("stats", text)
        self.assertIn("utterance", text)
        self.assertTrue(ended)
        self.assertEqual(self.upstream.app[FRAMES], list(range(6)))
        self.assertIn({"type": "stop"}, self.upstream.app[MESSAGES])
        diagnostics = await (await self.client.get("/api/diagnostics")).json()
        self.assertEqual(diagnostics["input_samples"], 12 * 2400)
        self.assertEqual(diagnostics["signal_samples"], 12 * 2400)
        self.assertAlmostEqual(diagnostics["input_peak_max"], 1000 / 32768, places=6)
        self.assertEqual(diagnostics["output_frames"], 6)
        self.assertGreater(diagnostics["voiced_frames"], 0)
        self.assertGreater(diagnostics["text_tokens"], 0)
        self.assertNotIn("pcm", diagnostics)

    async def test_silent_microphone_is_visible_in_diagnostics(self):
        socket = await self.client.ws_connect("/ws")
        await socket.send_json({"type": "start"})
        await self.receive_json(socket, "ready")
        await socket.send_bytes(b"\x01" + bytes(4800 * 2))
        while True:
            message = await socket.receive(timeout=3)
            if message.type == WSMsgType.BINARY:
                break
        diagnostics = await (await self.client.get("/api/diagnostics")).json()
        self.assertEqual(diagnostics["input_samples"], 4800)
        self.assertEqual(diagnostics["signal_samples"], 0)
        self.assertEqual(diagnostics["input_rms"], 0)
        self.assertEqual(diagnostics["input_peak_max"], 0)
        await socket.close()

    async def test_disconnect_releases_upstream(self):
        socket = await self.client.ws_connect("/ws")
        await socket.send_json({"type": "start"})
        await self.receive_json(socket, "ready")
        await socket.close()
        # A second connection must be accepted instead of being left permanently busy.
        for _ in range(20):
            health = await (await self.client.get("/api/health")).json()
            if not health["live"]:
                break
            await asyncio.sleep(0.01)
        self.assertFalse(health["live"])

    async def test_rejects_cross_site_websocket(self):
        with self.assertRaises(WSServerHandshakeError) as result:
            await self.client.ws_connect("/ws", origin="https://unrelated.example")
        self.assertEqual(result.exception.status, 403)

    async def test_same_origin_websocket(self):
        socket = await self.client.ws_connect("/ws", origin=str(self.client.make_url("/")).rstrip("/"))
        await socket.send_json({"type": "start"})
        self.assertIn("seed", await self.receive_json(socket, "ready"))
        await socket.close()

    async def test_public_origin_behind_tls_proxy(self):
        public_origin = "https://audio.example"
        client = TestClient(TestServer(create_app(
            str(self.upstream.make_url("/ws")).replace("http:", "ws:"), public_origin
        )))
        await client.start_server()
        try:
            # The browser uses HTTPS; the TLS proxy connects to this server over HTTP.
            socket = await client.ws_connect("/ws", origin=public_origin)
            await socket.send_json({"type": "start", "seed": 73})
            self.assertEqual((await self.receive_json(socket, "ready"))["seed"], 73)
            await socket.close()
            with self.assertRaises(WSServerHandshakeError) as result:
                await client.ws_connect(
                    "/ws", origin="https://unrelated.example",
                    headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "unrelated.example"},
                )
            self.assertEqual(result.exception.status, 403)
        finally:
            await client.close()

    async def test_upstream_unavailable(self):
        await self.upstream.close()
        response = await self.client.get("/api/health")
        self.assertEqual(response.status, 502)
        socket = await self.client.ws_connect("/ws")
        error = await self.receive_json(socket, "error")
        self.assertIn("Inference connection failed", error["message"])
        self.assertEqual((await socket.receive()).type, WSMsgType.CLOSE)

    async def test_busy_error_is_forwarded(self):
        self.upstream.app[STATE]["live"] = True
        socket = await self.client.ws_connect("/ws")
        await socket.send_json({"type": "start"})
        error = await self.receive_json(socket, "error")
        self.assertEqual(error["message"], "another conversation is running")
        await socket.close()


class ConfigurationTests(unittest.TestCase):
    def test_public_origin_validation(self):
        self.assertEqual(validate_public_origin(""), "")
        self.assertEqual(validate_public_origin("https://audio.example:8443/"), "https://audio.example:8443")
        for value in [
            "https://audio.example/path", "https://user:secret@audio.example", "ws://audio.example",
            "https://audio.example?query=1", "https://audio.example#fragment", "https://audio.example:bad",
        ]:
            with self.assertRaises(ValueError):
                validate_public_origin(value)

    def test_prefixed_https_metadata(self):
        self.assertEqual(
            api_url("wss://host.example/awaken/ws", "voices"), "https://host.example/awaken/voices"
        )

    def test_reject_invalid_urls(self):
        for url in [
            "http://localhost/ws",
            "ws:///ws",
            "ws://user:secret@localhost/ws",
            "ws://localhost/ws?key=1",
        ]:
            with self.assertRaises(ValueError):
                validate_upstream(url)


if __name__ == "__main__":
    unittest.main()
