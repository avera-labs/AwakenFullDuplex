"""Minimal WebSocket client: stream a user wav to a running server in real time, record the reply.

    python scripts/ws_client.py --url ws://localhost:8998/ws --wav user.wav --out reply/ [--prompt "..."]

Writes ``stereo.wav`` (left: what was sent, right: what came back, on the frame grid) and prints
the utterances and timing stats the server reports. Audio is sent at real-time pace in 100 ms
chunks, so the run takes as long as the wav plus the tail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

from awaken_chat.frame import SAMPLES_PER_FRAME_IN, SR_IN
from awaken_chat.inference.offline import load_user_track
from awaken_chat.server.protocol import AUDIO_FRAME, MIC_PCM, SILENT_FRAME


async def run(url: str, wav: str, out: str, prompt: str, voice: str, preset: str, seed: int | None,
              tail_s: float, chunk_ms: int) -> None:
    track = load_user_track(wav)
    track = np.concatenate([track, np.zeros(int(tail_s * SR_IN), np.float32)])
    chunk = SR_IN * chunk_ms // 1000
    reply: dict[int, np.ndarray] = {}
    utterances: list[dict] = []
    last_stats: dict = {}
    ready = asyncio.Event()
    done = asyncio.Event()

    async with websockets.connect(url, max_size=None) as ws:
        start = {"type": "start", "prompt": prompt, "voice": voice, "preset": preset}
        if seed is not None:
            start["seed"] = seed
        await ws.send(json.dumps(start))

        async def receiver() -> None:
            nonlocal last_stats
            async for msg in ws:
                if isinstance(msg, bytes):
                    kind, frame = struct.unpack_from("<BI", msg)
                    if kind == AUDIO_FRAME:
                        reply[frame] = np.frombuffer(msg[5:], dtype="<i2").astype(np.float32) / 32768.0
                    elif kind == SILENT_FRAME:
                        reply.setdefault(frame, None)
                    continue
                m = json.loads(msg)
                t = m.get("type")
                if t == "status":
                    print("status:", m["message"], flush=True)
                elif t == "ready":
                    print("ready: seed", m["seed"], flush=True)
                    ready.set()
                elif t == "utterance":
                    utterances.append(m)
                    print(f"[{m['t0']:6.1f}-{m['t1']:6.1f}] {m['text']}", flush=True)
                elif t == "stats":
                    last_stats = m
                elif t == "error":
                    print("error:", m["message"], flush=True)
                    ready.set()  # unblock the sender so the run ends
                elif t == "end":
                    print("end at frame", m["frames"], flush=True)
                    done.set()
                    return

        recv_task = asyncio.create_task(receiver())
        await ready.wait()  # the model may still be loading on a cold server
        t0 = time.perf_counter()
        for i in range(0, len(track), chunk):
            pcm = (np.clip(track[i:i + chunk], -1, 1) * 32767).astype("<i2").tobytes()
            await ws.send(bytes([MIC_PCM]) + pcm)
            target = t0 + (i + chunk) / SR_IN
            await asyncio.sleep(max(0.0, target - time.perf_counter()))
        await asyncio.sleep(1.0)
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(done.wait(), timeout=10)
        except asyncio.TimeoutError:
            pass
        recv_task.cancel()

    n_frames = (max(reply) + 1) if reply else 0
    agent = np.zeros(max(len(track), n_frames * SAMPLES_PER_FRAME_IN), np.float32)
    for f, pcm in reply.items():
        if pcm is not None:
            agent[f * SAMPLES_PER_FRAME_IN:f * SAMPLES_PER_FRAME_IN + len(pcm)] = pcm
    stereo = np.zeros((len(agent), 2), np.float32)
    stereo[: len(track), 0] = track
    stereo[:, 1] = agent
    Path(out).mkdir(parents=True, exist_ok=True)
    sf.write(str(Path(out) / "stereo.wav"), stereo, SR_IN)
    (Path(out) / "utterances.json").write_text(json.dumps(utterances, indent=1))
    voiced = sum(1 for v in reply.values() if v is not None)
    print(f"frames received {len(reply)} (voiced {voiced}); stats {last_stats}")
    print(f"-> {out}/stereo.wav")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8998/ws")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--voice", default="")
    ap.add_argument("--preset", default="live")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tail-s", type=float, default=6.0)
    ap.add_argument("--chunk-ms", type=int, default=100)
    a = ap.parse_args()
    asyncio.run(run(a.url, a.wav, a.out, a.prompt, a.voice, a.preset, a.seed, a.tail_s, a.chunk_ms))


if __name__ == "__main__":
    main()
