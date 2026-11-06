# Awaken Full Duplex

A standalone browser playground for the Awaken inference server. It uses the existing WebSocket
protocol and runs independently of the model, so the web client can run on a CPU. The implementation
and tests are in this directory.

## Run the playground

From the **repository root**, run the following with Python 3.10 or later and uv:

```bash
uv run --project web_client python web_client/server.py \
  --upstream ws://127.0.0.1:8998/ws
```

Open <http://localhost:8080>. The only runtime dependency is aiohttp. The server serves the frontend
files directly, with no build step.

To run without uv, install aiohttp in a separate virtual environment:

```bash
pip install 'aiohttp>=3.11,<4'
python web_client/server.py
```

For an inference server on a remote GPU host, forward its port over SSH:

```bash
ssh -N -L 8998:127.0.0.1:8998 user@gpu-host
# Run the web server command above in a separate terminal.
```

You can also point the web server directly at a remote upstream with
`--upstream ws://host:8998/ws` or `--upstream wss://host/ws`. The browser connects to the web
server's same-origin `/ws` endpoint and fetches `/health` and `/voices` through the proxy on
page load. This avoids configuring CORS on the inference server. Upstream path prefixes are supported.
For example, `wss://host/awaken/ws` maps to `https://host/awaken/health` for health checks.

The web server listens on `127.0.0.1:8080` by default. To allow access from another device, use
`--host 0.0.0.0 --port 8080`. The microphone and AudioWorklet require **HTTPS or localhost**. To
serve HTTPS directly, use a certificate trusted by the browser:

```bash
uv run --project web_client python web_client/server.py \
  --host 0.0.0.0 --ssl-cert certs/cert.pem --ssl-key certs/key.pem \
  --upstream ws://127.0.0.1:8998/ws
```

The web server is intended for development and has no login authentication. It rejects cross-site
WebSocket connections. Behind a TLS-terminating reverse proxy such as the Runpod HTTPS proxy, set
`--public-origin` to the browser-facing origin:

```bash
uv run --project web_client python web_client/server.py \
  --host 0.0.0.0 --upstream ws://127.0.0.1:8998/ws \
  --public-origin https://YOUR-POD-ID-8080.proxy.runpod.net
```

The configured origin must match the browser's scheme, host, and port exactly. The server ignores
`X-Forwarded-*` headers when checking the origin, and local same-origin access remains available.
A Runpod public port may differ from the container port. Use the proxy URL or mapped TCP port listed
in the Runpod console.

## Use the playground

- **Live microphone.** Choose an input device and grant permission when starting. Once the server
  sends `ready`, the AudioWorklet continuously sends 100 ms PCM chunks while the browser plays
  the model's replies. This mode uses neither voice activity detection nor push-to-talk. Echo
  cancellation and automatic gain are on by default and can be turned off. Headphones are
  recommended. The UI shows the active input device and reports muted input, paused audio, or
  missing audio. Stopping sends about one second of silence to let the streaming decoder finish.
- **Audio files.** Select a WAV, MP3, or other file that the browser can decode. The browser
  resamples it to 24 kHz, averages the channels to mono, and streams it to the same inference
  endpoint at real-time speed. After the file ends, it sends a configurable tail of silence
  (six seconds by default) for the model to reply, then ends the session. This mode works without
  microphone access. Keep the tab in the foreground for consistent send timing.
- **Session settings.** Set an additional system prompt, a voice, the `live` or `bench` preset,
  a random seed, JSON overrides for supported `DecodeParams` fields, or the playback buffer
  (200 ms by default). The client selects `preset-female` when available and otherwise uses the
  server default. The debug panel shows the actual settings reported in `ready`.
- **Recording.** Recording is on by default. After a session ends or disconnects, download a
  24 kHz stereo PCM16 WAV with **the user on the left and the model on the right**. The recording
  aligns the input sample positions with the server's frame numbers and preserves overlaps,
  silences, and gaps for missing frames. It follows the model timeline, excluding network and
  speaker playback delay. Muting playback has no effect on the recording. Debug JSON is available
  with recording turned off.
- **Playback at session end.** The player waits for queued speech to finish before closing. If
  the browser suspends audio, click **Resume audio** to play the rest. Starting a new session
  stops playback from the previous one.
- **Debugging.** Inspect streaming model text, finalized utterances, per-frame text and audio
  classes, raw JSON events, transport counters, decoder settings, microphone settings, and server
  health. The UI shows the latest 300 events and 200 frames. The JSON export keeps up to 10,000
  events and reports how many were dropped. The text feed shows the model's output.

Recordings stay in browser memory. Download a recording before refreshing or starting another
session. Audio files are limited to 100 MB and 300 seconds. The inference server's default session
limit is also 300 seconds, including trailing silence. The server supports one session at a time.
If the server is busy, the connection drops, or inference fails, the UI shows the error and you can
still download any data already received.

## Performance metrics

| Metric | What it measures |
| --- | --- |
| First audio / First text | Browser time from receiving `ready` to receiving the first non-silent audio packet or content text. This includes time waiting for user input and server batching. |
| Audio frame latency P50 / P95 | Time from sending the last samples of a 200 ms user frame with `WebSocket.send` to receiving the matching model audio frame, over the last 100 frames. This includes network time, server processing, and decoding. Frame capture time and speaker playback delay are outside this measurement. |
| Model step / Ear / Detokenizer | P50 / P95 in milliseconds from the server's `stats` events, calculated over the last 50 samples for each stage. Each stage reports its own percentiles, so adding them does not give an end-to-end percentile. |
| Server lag | `lag_frames`, where each frame is 200 ms. |
| Playback queue / Rebuffers / Missing frames | Remaining scheduled playback time in Web Audio, the number of times the playback clock restarted after late audio, and gaps in server frame numbers. Silent frames advance the playback clock. |
| Traffic / WS buffered | Application-level bytes sent and received in the session, excluding transport overhead, and the browser's WebSocket `bufferedAmount`. The client stops when pending data exceeds one second of PCM to avoid sending stale microphone audio. |

At the end of a session, the client pads the recording to a 200 ms frame boundary and sets the
expected output frame count. It then sends an extra 100 ms of silence as lookahead for the streaming
ear to release the final frame. Once the expected frames arrive, it sends `stop`. The extra
silence is excluded from the recording and the expected frame count. In debug JSON,
`input_samples` counts the recorded input samples and `sent_samples` includes the extra
`drain_context_samples`. If inference or the network falls behind, the client can time out and
preserve the results received so far. Browser sleep, background throttling, and changes to the audio
device can affect real-time timing. The event log records observable issues.

`/api/diagnostics` reports input sample counts, audio levels, model output frames, and text
token counts for the most recent proxy connection. Use it to tell whether the microphone is silent
or the model is still listening. The endpoint keeps counters and status only. Before closing the
WebSocket, the inference server waits for the final `utterance`, `frames`, and `end`
messages to be sent.

## Tests without a model

Testing requires Node.js 20 or later, npm, and uv. From the repository root, run:

```bash
uv sync --project web_client
npm ci --prefix web_client
npm exec --prefix web_client -- playwright install chromium
npm test --prefix web_client
```

`npm test` runs audio unit tests, HTTP and WebSocket proxy integration tests, and Chromium
end-to-end tests. The browser tests start a dedicated web server and protocol simulator and use a
virtual microphone and synthetic WAV files. They run without a physical microphone, model weights,
or GPU inference server. The default test port is 18765. Set `E2E_PORT` to override it.

The tests cover file resampling and downmixing, live bidirectional audio, session settings,
left and right channels and silent intervals in recordings, WAV and JSON downloads, recording
opt-out, busy and failing inference servers, upstream connection failures, cancellation while
waiting for microphone permission, and mobile layout. On failure, screenshots and traces go in
the ignored `test-results/` directory.

To run the suites individually:

```bash
npm run test:unit --prefix web_client
npm run test:proxy --prefix web_client
npm run test:e2e --prefix web_client
```

For manual browser testing, start the simulator and web server in separate terminals:

```bash
# Terminal 1: Generate a sine wave and fixed test text.
uv run --project web_client python web_client/tests/mock_inference.py

# Terminal 2
uv run --project web_client python web_client/server.py --upstream ws://127.0.0.1:8999/ws
```

The simulator supports `--busy`, `--fail-after 3`, and `--max-frames 40`. It implements the
protocol defined by `protocol.py`, `app.py`, and `session_runner.py` in
`awaken_chat/server/`, with fixed performance metrics for tests. Connect to a real inference
server to evaluate voice quality, model behavior, and GPU performance.

## Files and protocol

```text
server.py                 Static assets, same-origin HTTP and WebSocket proxy, optional HTTPS
static/app.js             Session lifecycle, live and file input, debugging UI
static/audio.js           Web Audio, file decoding, continuous playback, device cleanup
static/capture-worklet.js Microphone capture on the audio thread
static/core.js            PCM protocol, playback clock, frame-aligned stereo WAV
tests/                    Unit, proxy integration, and browser tests plus the protocol simulator
```

The browser sends a JSON `start` message, waits for `ready`, and then sends binary packets
consisting of `0x01 + PCM16 LE`. Each 200 ms server frame is
`0x02 + uint32 LE frame number + 4800 PCM16 samples` or
`0x03 + uint32 LE frame number` for a silent frame. JSON messages carry `status`,
`frames`, `utterance`, `stats`, `error`, and `end` events. The browser sends
periodic JSON `ping` messages to keep the connection alive.

See [AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletProcessor) and
[getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia) for the
browser APIs. See [Playwright webServer](https://playwright.dev/docs/test-webserver) and
[WebRTC virtual media devices](https://webrtc.github.io/webrtc-org/testing/) for the test setup.
