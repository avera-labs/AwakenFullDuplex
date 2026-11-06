"""Start the real playground against an isolated protocol simulator, only for browser E2E."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import UPSTREAM, create_app
from tests.mock_inference import FRAMES, MESSAGES, STATE, create_mock


def create_e2e_app() -> web.Application:
    app = create_app()
    simulator = create_mock()

    async def backend_lifecycle(application):
        runner = web.AppRunner(simulator)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        application[UPSTREAM] = f"ws://127.0.0.1:{port}/ws"
        yield
        await runner.cleanup()

    async def scenario(request):
        settings = await request.json()
        simulator[STATE].update(
            {"live": False, "fail_after": None, "ready_delay": 0.01, "offline": False,
             "holdback_samples": 0, **settings}
        )
        simulator[FRAMES].clear()
        simulator[MESSAGES].clear()
        return web.json_response({"ok": True})

    # This route exists only in the E2E harness, never in the shipped service.
    app.router.add_post("/_test/scenario", scenario)
    app.cleanup_ctx.append(backend_lifecycle)
    return app


if __name__ == "__main__":
    web.run_app(create_e2e_app(), host="127.0.0.1", port=int(os.environ.get("E2E_PORT", "18765")))
