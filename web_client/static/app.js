import {
  CHUNK_SAMPLES,
  FRAME_SAMPLES,
  SAMPLE_RATE,
  Recording,
  decodeAudio,
  encodeMic,
  rms,
} from "./core.js";
import { AudioIO } from "./audio.js";

const $ = (id) => document.getElementById(id);
const encoder = new TextEncoder();
const decodeFields = new Set([
  "text_temperature",
  "text_top_k",
  "text_temperature_burst",
  "text_top_k_burst",
  "text_rep_penalty",
  "text_rep_window",
  "audio_temperature",
  "audio_top_k",
  "max_silent_run",
  "max_utterance_s",
]);
let active = null;
let last = null;
let health = null;
let previewUrl = null;
let refreshId = 0;
const waves = { user: new Float32Array(0), agent: new Float32Array(0) };

function notice(message = "", info = false) {
  $("notice").textContent = message;
  $("notice").hidden = !message;
  $("notice").classList.toggle("info", info);
}

function connection(text, live = false) {
  $("connection").replaceChildren(
    Object.assign(document.createElement("i"), {}),
    document.createTextNode(text),
  );
  $("connection").classList.toggle("active", live);
}

function log(s, type, data, level = "") {
  const entry = {
    elapsed_ms: Math.round(performance.now() - s.createdAt),
    type,
    data,
  };
  s.events.push(entry);
  if (s.events.length > 10000) {
    s.events.shift();
    s.droppedEvents++;
  }
  const row = document.createElement("div");
  row.className = `event ${level}`;
  const time = document.createElement("time");
  time.textContent = `${(entry.elapsed_ms / 1000).toFixed(3)} s`;
  const kind = document.createElement("b");
  kind.textContent = type;
  const content = document.createElement("span");
  content.textContent = typeof data === "string" ? data : JSON.stringify(data);
  row.append(time, kind, content);
  if (!$("events").querySelector(".event")) $("events").replaceChildren();
  $("events").append(row);
  while ($("events").children.length > 300) $("events").firstChild.remove();
  $("event-count").textContent = s.events.length;
  if ($("autoscroll").checked)
    $("tab-events").scrollTop = $("tab-events").scrollHeight;
}

async function refresh() {
  const id = ++refreshId;
  $("refresh").disabled = true;
  try {
    const fetchJson = async (path) => {
      const response = await fetch(path, {
        signal: AbortSignal.timeout(12000),
      });
      const body = await response.json();
      if (!response.ok)
        throw new Error(body.error || `HTTP ${response.status}`);
      return body;
    };
    const results = await Promise.allSettled([
      fetchJson("/api/config"),
      fetchJson("/api/health"),
      fetchJson("/api/voices"),
    ]);
    if (id !== refreshId) return;
    if (results[0].status === "fulfilled")
      $("upstream").textContent = results[0].value.upstream;
    if (results[1].status === "fulfilled") {
      health = results[1].value;
      $("health-json").textContent = JSON.stringify(health, null, 2);
      $("health-status").textContent = health.live
        ? "Inference server busy"
        : health.ready
          ? "Inference server ready"
          : "Inference server online · Waiting for model";
      $("health-dot").className = `dot ${health.live ? "" : "good"}`;
    } else {
      health = { error: results[1].reason.message };
      $("health-status").textContent = "Inference server unreachable";
      $("health-dot").className = "dot bad";
      $("health-json").textContent = JSON.stringify(health, null, 2);
    }
    if (results[2].status === "fulfilled" && !active) {
      const selected = $("voice").value;
      $("voice").replaceChildren(new Option("Server default", ""));
      for (const voice of results[2].value.voices || [])
        $("voice").add(new Option(voice, voice));
      if ([...$("voice").options].some((option) => option.value === selected))
        $("voice").value = selected;
    }
  } catch (error) {
    $("health-status").textContent = error.message;
  } finally {
    if (id === refreshId) $("refresh").disabled = false;
  }
}

async function refreshMicrophones() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const select = $("microphone-device");
  const selected = select.value;
  try {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter(
      (device) => device.kind === "audioinput" && device.deviceId,
    );
    select.replaceChildren(new Option("System default microphone", ""));
    for (const [index, device] of devices.entries())
      select.add(
        new Option(device.label || `Microphone ${index + 1}`, device.deviceId),
      );
    if ([...select.options].some((option) => option.value === selected))
      select.value = selected;
  } catch (error) {
    $("microphone-status").textContent =
      `Cannot list microphones: ${error.message}`;
  }
}

function options() {
  const mode = document.querySelector("input[name=mode]:checked").value;
  const file = $("audio-file").files[0];
  if (mode === "file" && !file) throw new Error("Choose an audio file first.");
  const seed = $("seed").value === "" ? undefined : Number($("seed").value);
  if (
    seed !== undefined &&
    (!Number.isInteger(seed) || seed < 0 || seed >= 2 ** 30)
  ) {
    throw new Error("Random seed must be an integer from 0 to 1073741823.");
  }
  const decode = JSON.parse($("decode").value.trim() || "{}");
  if (!decode || Array.isArray(decode) || typeof decode !== "object")
    throw new Error("Decode overrides must be a JSON object.");
  for (const [key, value] of Object.entries(decode)) {
    if (
      !decodeFields.has(key) ||
      typeof value !== "number" ||
      !Number.isFinite(value)
    ) {
      throw new Error(
        `Invalid decode parameter: ${key}. Use a supported field with a finite numeric value.`,
      );
    }
    if (
      (key.includes("top_k") ||
        key === "text_rep_window" ||
        key === "max_silent_run") &&
      !Number.isInteger(value)
    ) {
      throw new Error(`${key} must be an integer.`);
    }
    if (
      (key === "text_rep_penalty" || key === "max_utterance_s") &&
      value <= 0
    ) {
      throw new Error(`${key} must be greater than 0.`);
    }
  }
  const buffer = Number($("buffer").value);
  const tail = Number($("tail").value);
  if (!Number.isFinite(buffer) || buffer < 40 || buffer > 2000)
    throw new Error("Playback buffer must be between 40 and 2000 ms.");
  if (!Number.isFinite(tail) || tail < 0 || tail > 60)
    throw new Error("Trailing silence must be between 0 and 60 seconds.");
  return {
    microphoneDevice: $("microphone-device").value,
    autoGainControl: $("auto-gain").checked,
    mode,
    file,
    buffer,
    tail,
    record: $("record").checked,
    echo: $("echo").checked,
    start: {
      type: "start",
      prompt: $("prompt").value,
      voice: $("voice").value,
      preset: $("preset").value,
      decode,
      ...(seed === undefined ? {} : { seed }),
    },
  };
}

function updateDownloads(s) {
  $("download-wav").disabled =
    !s?.closed || !s.recording.enabled || !s.recording.length;
  $("download-json").disabled = !s?.closed;
}

function resetDisplay(s) {
  $("transcript").replaceChildren(
    Object.assign(document.createElement("p"), {
      className: "empty",
      textContent: "Waiting for a reply…",
    }),
  );
  $("frames").replaceChildren();
  $("events").replaceChildren();
  $("ready-json").textContent = "Waiting for ready…";
  $("mic-json").textContent = "—";
  $("event-count").textContent = "0";
  $("token-state").textContent = "WAITING";
  $("first-audio").innerHTML = "—<small>ms</small>";
  $("frame-latency").innerHTML = "—<small>ms</small>";
  $("server-lag").innerHTML = "—<small>frames</small>";
  $("mode-label").textContent = s.options.mode === "mic" ? "LIVE" : "FILE";
  $("recording-status").classList.toggle("recording", s.recording.enabled);
  $("recording-status").replaceChildren(
    document.createElement("i"),
    document.createTextNode(
      s.recording.enabled
        ? "Recording enabled · Waiting for input"
        : "Recording disabled",
    ),
  );
  $("recording-status").firstChild.className = "dot";
  $("input-status").textContent = "Preparing audio…";
  $("output-status").textContent = "Waiting for model";
  $("resume-audio").hidden = true;
  waves.user = waves.agent = new Float32Array(0);
  for (const stage of ["step", "ear", "detok"])
    for (const p of ["p50", "p95"]) $(`${stage}-${p}`).textContent = "—";
  updateDownloads(s);
}

async function start() {
  if (active) return;
  let opts;
  try {
    opts = options();
  } catch (error) {
    notice(error.message);
    return;
  }
  notice();
  const s = {
    options: opts,
    createdAt: performance.now(),
    wallTime: new Date().toISOString(),
    phase: "preparing",
    closed: false,
    recording: new Recording(opts.record),
    events: [],
    droppedEvents: 0,
    utterances: [],
    frames: [],
    perf: [],
    latencies: [],
    inputFrameTimes: [],
    sentSamples: 0,
    drainTargetSamples: null,
    drainContextSamples: 0,
    sentBytes: 0,
    receivedBytes: 0,
    receivedFrames: 0,
    lastAudioFrame: -1,
    silentFrames: 0,
    textParts: [],
    health,
    readyAt: null,
    firstAudio: null,
    firstText: null,
    stats: null,
    lastInputAt: null,
    inputPeakMax: 0,
    signalSamples: 0,
    status: "preparing",
  };
  active = s;
  last = s;
  document.body.classList.add("running");
  $("settings").disabled = true;
  $("stop").disabled = false;
  $("stop").textContent = "■ End and save recording";
  resetDisplay(s);
  connection("Preparing audio");
  log(s, "session", {
    mode: opts.mode,
    recording: opts.record,
    file: opts.file?.name,
  });
  s.audio = new AudioIO(
    opts.buffer / 1000,
    (samples) => {
      if (s.phase === "running" || s.phase === "flushing")
        sendAudio(s, samples);
    },
    (state) => {
      if (s.closed) {
        if (last === s && !active)
          $("resume-audio").hidden = state === "running" || state === "closed";
        return;
      }
      log(s, "audio-context", state, state === "running" ? "" : "warn");
      $("resume-audio").hidden = state === "running";
    },
  );
  try {
    await s.audio.open();
    if (s.closed) return;
    s.audio.mute($("mute").checked);
    if (opts.mode === "mic") {
      const settings = await s.audio.microphone(
        opts.echo,
        opts.microphoneDevice,
        opts.autoGainControl,
      );
      if (s.closed) return;
      const micTrack = s.audio.stream.getAudioTracks()[0];
      s.micSettings = { ...settings, label: micTrack.label };
      $("mic-json").textContent = JSON.stringify(s.micSettings, null, 2);
      log(s, "microphone", s.micSettings);
      refreshMicrophones();
      micTrack.onmute = () =>
        log(
          s,
          "microphone-muted",
          "The browser or system paused microphone input.",
          "warn",
        );
      micTrack.onunmute = () =>
        log(s, "microphone-unmuted", "Microphone input resumed.");
      for (const track of s.audio.stream.getAudioTracks())
        track.onended = () => {
          if (s.phase === "running")
            fail(
              s,
              "Microphone disconnected. Check the device and start again.",
            );
        };
    } else {
      s.fileSamples = await s.audio.decodeFile(opts.file);
      if (s.closed) return;
      if (!s.fileSamples.length) throw new Error("The audio file is empty.");
      log(s, "file", {
        name: opts.file.name,
        samples: s.fileSamples.length,
        duration_s: s.fileSamples.length / SAMPLE_RATE,
        tail_s: opts.tail,
      });
      if (s.fileSamples.length / SAMPLE_RATE + opts.tail > 300) {
        log(
          s,
          "warning",
          "The file and trailing silence exceed the default 300-second session limit. The server may end the session early.",
          "warn",
        );
      }
    }
    if (s.closed) return;
    connect(s);
  } catch (error) {
    if (!s.closed) fail(s, `Audio setup failed: ${error.message}`);
  }
}

function connect(s) {
  const url = new URL("/ws", location.href);
  url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(url);
  s.ws = ws;
  ws.binaryType = "arraybuffer";
  s.phase = "connecting";
  connection("Connecting to inference server");
  s.connectTimer = setTimeout(
    () => fail(s, "WebSocket connection timed out."),
    15000,
  );
  ws.onopen = () => {
    if (s.closed) {
      ws.close();
      return;
    }
    clearTimeout(s.connectTimer);
    s.phase = "waiting";
    connection("Waiting for model readiness");
    sendJson(s, s.options.start);
    s.pingTimer = setInterval(
      () => sendJson(s, { type: "ping" }, false),
      15000,
    );
  };
  ws.onmessage = ({ data }) => {
    if (s.closed) return;
    try {
      s.receivedBytes +=
        typeof data === "string"
          ? encoder.encode(data).length
          : data.byteLength;
      if (typeof data !== "string") {
        receiveAudio(s, decodeAudio(data));
        return;
      }
      const message = JSON.parse(data);
      log(
        s,
        message.type || "message",
        message,
        message.type === "error" ? "error" : "",
      );
      receiveJson(s, message);
    } catch (error) {
      fail(s, `Protocol or playback error: ${error.message}`);
    }
  };
  ws.onerror = () => {
    if (!s.closed)
      fail(
        s,
        "WebSocket connection failed. Check the web server and inference server address.",
      );
  };
  ws.onclose = ({ code, reason }) => {
    if (s.closed) return;
    log(s, "socket-close", { code, reason });
    if (s.phase !== "stopping" && s.phase !== "ended") {
      notice(
        `Inference connection closed (${code}). Received audio and debug data are still available to download.`,
      );
      finish(s, "disconnected");
    } else finish(s, "completed");
  };
}

function sendJson(s, message, verbose = true) {
  if (s.closed || s.ws?.readyState !== WebSocket.OPEN) return;
  const value = JSON.stringify(message);
  s.ws.send(value);
  s.sentBytes += encoder.encode(value).length;
  if (verbose) log(s, `send:${message.type}`, message);
}

function sendAudio(s, samples, contextOnly = false) {
  if (s.closed || s.ws?.readyState !== WebSocket.OPEN || !samples.length)
    return;
  // Do not accumulate seconds of stale microphone audio on a stalled connection.
  if (s.ws.bufferedAmount > SAMPLE_RATE * 2) {
    fail(
      s,
      "Outgoing audio fell more than 1 second behind. The connection was stopped. Check your network and try again.",
    );
    return;
  }
  const packet = encodeMic(samples);
  s.ws.send(packet);
  s.lastInputAt = performance.now();
  const level = rms(samples);
  for (const sample of samples)
    s.inputPeakMax = Math.max(s.inputPeakMax, Math.abs(sample));
  if (level >= 0.001) s.signalSamples += samples.length;
  s.sentBytes += packet.byteLength;
  const previous = Math.floor(s.sentSamples / FRAME_SAMPLES);
  s.sentSamples += samples.length;
  const completed = Math.floor(s.sentSamples / FRAME_SAMPLES);
  for (let frame = previous; frame < completed; frame++)
    s.inputFrameTimes[frame] = performance.now();
  if (contextOnly) s.drainContextSamples += samples.length;
  else s.recording.addUser(samples);
  waves.user = samples;
  if (!s.lastInputLog || performance.now() - s.lastInputLog > 1000) {
    log(s, "input:pcm", {
      samples_sent: s.sentSamples,
      buffered_bytes: s.ws.bufferedAmount,
      rms: Number(level.toFixed(6)),
    });
    s.lastInputLog = performance.now();
  }
}

function receiveAudio(s, audio) {
  if (s.readyAt === null) throw new Error("Audio arrived before ready");
  // The server caps sessions to 300 s by default. Bound malformed indices before allocation/export.
  if (audio.frame > 108000)
    throw new Error("Audio frame index exceeds the 6-hour client limit");
  s.audio.play(audio);
  s.recording.addAgent(audio.frame, audio.samples, audio.silent);
  s.lastAudioFrame = audio.frame;
  s.receivedFrames++;
  s.silentFrames += Number(audio.silent);
  const sentAt = s.inputFrameTimes[audio.frame];
  if (sentAt !== undefined) {
    s.latencies.push(performance.now() - sentAt);
    if (s.latencies.length > 100) s.latencies.shift();
  }
  if (!audio.silent && s.firstAudio === null) {
    s.firstAudio = performance.now() - s.readyAt;
    log(s, "first-audio", { since_ready_ms: Math.round(s.firstAudio) });
  }
  waves.agent = audio.samples;
  if (audio.frame % 5 === 0)
    log(s, "output:pcm", {
      frame: audio.frame,
      silent: audio.silent,
      playback_queue_ms: Math.round(s.audio.queueMs),
      underruns: s.audio.clock.underruns,
    });
  maybeDrained(s);
}

function receiveJson(s, message) {
  if (message.type === "status") {
    connection(message.message);
    return;
  }
  if (message.type === "ready") {
    if (s.readyAt !== null) throw new Error("Duplicate ready message");
    s.ready = message;
    $("ready-json").textContent = JSON.stringify(message, null, 2);
    s.readyAt = performance.now();
    s.phase = "running";
    connection("Conversation active", true);
    $("recording-status").lastChild.textContent = s.recording.enabled
      ? "Recording · User L / AI R"
      : "Recording disabled";
    $("input-status").textContent =
      s.options.mode === "mic"
        ? "Capturing microphone"
        : "Streaming audio file";
    if (s.options.mode === "mic") s.audio.startMicrophone();
    else feedFile(s);
  } else if (message.type === "frames") {
    for (const item of message.items) {
      s.frames.push(item);
      if (item.text !== null && item.text !== undefined) {
        if (s.firstText === null) s.firstText = performance.now() - s.readyAt;
        s.textParts.push({ frame: item.f, text: item.text });
      }
      $("token-state").textContent =
        `${item.text_class.toUpperCase()} · ${item.audio_class.join(" ")}`;
      const row = document.createElement("tr");
      for (const value of [
        item.f,
        `${item.t.toFixed(1)} s`,
        item.text_class,
        item.text ?? "—",
        item.audio_class.join(" · "),
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      $("frames").append(row);
    }
    while ($("frames").children.length > 200) $("frames").firstChild.remove();
    renderTranscript(s);
    if ($("autoscroll").checked)
      $("tab-frames").scrollTop = $("tab-frames").scrollHeight;
  } else if (message.type === "utterance") {
    s.utterances.push(message);
    renderTranscript(s);
  } else if (message.type === "stats") {
    s.stats = message;
    s.perf.push({
      elapsed_ms: Math.round(performance.now() - s.readyAt),
      ...message,
    });
    if (message.lag_frames > 10 && !s.warnedLag) {
      s.warnedLag = true;
      log(
        s,
        "warning",
        "Server backlog exceeds 2 seconds. Check the performance panel.",
        "warn",
      );
    }
  } else if (message.type === "error")
    fail(s, message.message || "The inference server reported an error.");
  else if (message.type === "end") {
    s.end = message;
    s.phase = "ended";
    finish(s, "completed");
  }
}

function renderTranscript(s) {
  const fragment = document.createDocumentFragment();
  let endFrame = -1;
  for (const utterance of s.utterances) {
    const line = document.createElement("p");
    line.className = "utterance";
    const time = document.createElement("time");
    time.textContent = `${utterance.t0.toFixed(1)}–${utterance.t1.toFixed(1)} s`;
    const text = document.createElement("span");
    text.textContent = utterance.text;
    line.append(time, text);
    fragment.append(line);
    endFrame = Math.max(endFrame, Math.round(utterance.t1 * 5) - 1);
  }
  const text = s.textParts
    .filter((part) => part.frame > endFrame)
    .map((part) => part.text)
    .join("");
  if (text) {
    const line = document.createElement("p");
    line.className = "partial";
    line.textContent = text;
    fragment.append(line);
  }
  if (fragment.childNodes.length) $("transcript").replaceChildren(fragment);
  if ($("autoscroll").checked)
    $("transcript").scrollTop = $("transcript").scrollHeight;
}

function feedFile(s) {
  const total = s.fileSamples.length + Math.round(s.options.tail * SAMPLE_RATE);
  const tick = () => {
    if (s.closed || s.phase !== "running") return;
    const remaining = total - s.sentSamples;
    if (remaining <= 0) {
      drain(s);
      return;
    }
    const target = s.readyAt + (s.sentSamples / SAMPLE_RATE) * 1000;
    const late = performance.now() - target;
    if (late > 1000) {
      fail(
        s,
        "Audio file streaming fell more than 1 second behind. Keep the page in the foreground and try again.",
      );
      return;
    }
    const count = Math.min(CHUNK_SAMPLES, remaining);
    const chunk = new Float32Array(count);
    if (s.sentSamples < s.fileSamples.length) {
      chunk.set(
        s.fileSamples.subarray(
          s.sentSamples,
          Math.min(s.sentSamples + count, s.fileSamples.length),
        ),
      );
    }
    sendAudio(s, chunk);
    $("input-status").textContent =
      s.sentSamples < s.fileSamples.length
        ? "Streaming audio file"
        : "Trailing silence · Waiting for reply";
    s.fileTimer = setTimeout(
      tick,
      Math.max(
        0,
        s.readyAt + (s.sentSamples / SAMPLE_RATE) * 1000 - performance.now(),
      ),
    );
  };
  tick();
}

async function stop() {
  const s = active;
  if (!s || s.closed) return;
  if (s.phase !== "running") {
    finish(s, "cancelled");
    return;
  }
  $("stop").disabled = true;
  $("stop").textContent = "Finishing…";
  clearTimeout(s.fileTimer);
  s.phase = "flushing";
  if (s.options.mode === "mic") {
    await s.audio.stopMicrophone();
    if (s.closed) return;
    // Keep the model clock moving long enough to flush the streaming codec's tail.
    let chunks = 10;
    const silence = () => {
      if (s.closed) return;
      if (chunks-- === 0) {
        drain(s);
        return;
      }
      sendAudio(s, new Float32Array(CHUNK_SAMPLES));
      s.fileTimer = setTimeout(silence, 100);
    };
    silence();
  } else drain(s);
}

function drain(s) {
  if (s.closed) return;
  s.phase = "draining";
  $("stop").disabled = true;
  $("stop").textContent = "Waiting for remaining frames…";
  $("input-status").textContent =
    "Input complete · Waiting for remaining frames";
  connection("Finishing", true);
  const remainder = s.sentSamples % FRAME_SAMPLES;
  if (remainder) sendAudio(s, new Float32Array(FRAME_SAMPLES - remainder));
  // Freeze the output target before providing the ear's final lookahead. Waiting on
  // the lookahead itself would create another unfinished frame and repeat the stall.
  s.drainTargetSamples = s.sentSamples;
  // StreamingEar needs ~80 ms of future samples (resample + mel/conv lookahead).
  // One 100 ms chunk releases the last user frame without extending the recording.
  sendAudio(s, new Float32Array(CHUNK_SAMPLES), true);
  if (s.closed) return;
  log(s, "drain", {
    expected_frames: s.drainTargetSamples / FRAME_SAMPLES,
    context_samples: s.drainContextSamples,
    received_frames: s.receivedFrames,
  });
  s.drainTimer = setTimeout(() => {
    log(
      s,
      "warning",
      "Timed out waiting for remaining frames. Keeping received data.",
      "warn",
    );
    notice(
      "Timed out waiting for remaining frames. Downloads include sent user audio and received AI audio.",
    );
    requestEnd(s);
  }, 15000);
  maybeDrained(s);
}

function maybeDrained(s) {
  if (
    s.phase === "draining" &&
    (s.lastAudioFrame + 1) * FRAME_SAMPLES >= s.drainTargetSamples
  )
    requestEnd(s);
}

function requestEnd(s) {
  if (s.closed || s.phase === "stopping") return;
  clearTimeout(s.drainTimer);
  s.phase = "stopping";
  sendJson(s, { type: "stop" });
  s.stopTimer = setTimeout(() => {
    log(
      s,
      "warning",
      "No end message received. Ending with the available data.",
      "warn",
    );
    finish(s, "completed-without-end");
  }, 7000);
}

function fail(s, message) {
  if (s.closed) return;
  notice(message);
  log(s, "error", message, "error");
  finish(s, "error");
}

function finish(s, status) {
  if (s.closed) return;
  s.closed = true;
  s.status = status;
  s.finishedAt = performance.now();
  for (const timer of ["connectTimer", "fileTimer", "drainTimer", "stopTimer"])
    clearTimeout(s[timer]);
  clearInterval(s.pingTimer);
  log(s, "finished", {
    status,
    sent_samples: s.sentSamples,
    received_frames: s.receivedFrames,
    silent_frames: s.silentFrames,
    recording_seconds: s.recording.length / SAMPLE_RATE,
  });
  if (s.ws && s.ws.readyState < WebSocket.CLOSING) s.ws.close();
  s.audio.stream?.getTracks().forEach((track) => track.stop());
  s.audio.capture?.port.postMessage("stop");
  // Wait for actual source.onended events: wall-clock timers keep running while
  // Web Audio is suspended, and a 10 s cap can discard an otherwise complete reply.
  const playRemaining =
    status.startsWith("completed") && s.audio.sources.size > 0;
  const closing = playRemaining ? s.audio.finishPlayback() : s.audio.close();
  closing
    .then(() => {
      if (last === s && !active) {
        $("output-status").textContent = "Playback finished";
        $("resume-audio").hidden = true;
      }
    })
    .catch((error) => {
      if (last === s && !active) notice(error.message);
    });
  s.fileSamples = null;
  connection(
    status.startsWith("completed")
      ? "Session ended"
      : status === "cancelled"
        ? "Session cancelled"
        : "Session interrupted",
  );
  $("settings").disabled = false;
  $("stop").disabled = true;
  $("input-status").textContent = "Input ended";
  $("output-status").textContent = playRemaining
    ? "Playing remaining audio"
    : "Playback finished";
  $("recording-status").classList.remove("recording");
  $("recording-status").lastChild.textContent =
    s.recording.enabled && s.recording.length
      ? "Recording ready · Download WAV"
      : "No recording saved";
  document.body.classList.remove("running");
  active = null;
  updateDownloads(s);
  refresh();
}

function metric(id, value, unit) {
  $(id).replaceChildren(
    document.createTextNode(value),
    Object.assign(document.createElement("small"), { textContent: unit }),
  );
}

function renderMicrophoneStatus(s, elapsed) {
  if (s.options.mode !== "mic" || s.closed || s.readyAt === null) return;
  const ctx = s.audio.context;
  const name = s.micSettings?.label || "Current microphone";
  const track = s.audio.stream?.getAudioTracks()[0];
  let status = `${name} · Receiving audio`;
  if (track?.muted)
    status = `${name} · Input muted by the browser or system. Check microphone permissions and device.`;
  else if (ctx?.state !== "running")
    status = `${name} · Audio paused. Click Resume audio.`;
  else if (performance.now() - (s.lastInputAt ?? s.readyAt) > 2000)
    status = `${name} · No audio data received. Check microphone permissions or choose another device.`;
  else if (elapsed >= 4 && !s.signalSamples)
    status = `${name} · No sound detected. Speak or choose another microphone, and check the system input volume.`;
  else if (!s.signalSamples) status = `${name} · Waiting for speech`;
  $("microphone-status").textContent = status;
  if (ctx?.state !== "running") $("resume-audio").hidden = false;
}

function renderMetrics() {
  const s = active || last;
  if (!s) return;
  const elapsed =
    s.readyAt === null
      ? 0
      : ((s.finishedAt || performance.now()) - s.readyAt) / 1000;
  $("duration").textContent = `${Math.floor(elapsed / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor(elapsed % 60)
    .toString()
    .padStart(2, "0")}`;
  $("input-duration").textContent =
    `${(s.sentSamples / SAMPLE_RATE).toFixed(1)} s sent`;
  $("output-duration").textContent =
    `${s.receivedFrames} frames · ${s.silentFrames} silent`;
  if (s.firstAudio !== null)
    metric("first-audio", Math.round(s.firstAudio), "ms");
  if (s.latencies.length) {
    const values = [...s.latencies].sort((a, b) => a - b);
    const p = (quantile) =>
      Math.round(
        values[
          Math.min(values.length - 1, Math.floor(values.length * quantile))
        ],
      );
    metric("frame-latency", `${p(0.5)} / ${p(0.95)}`, "ms");
  }
  if (s.stats) {
    metric("server-lag", s.stats.lag_frames, "frames");
    for (const [stage, key] of [
      ["step", "step_ms"],
      ["ear", "ear_ms"],
      ["detok", "detok_ms"],
    ]) {
      for (const p of ["p50", "p95"])
        $(`${stage}-${p}`).textContent =
          s.stats[key]?.[p] === undefined ? "—" : `${s.stats[key][p]} ms`;
    }
  }
  metric("playback-queue", Math.round(s.audio.queueMs), "ms");
  $("traffic").textContent =
    `↑ ${(s.sentBytes / 1024).toFixed(1)} KB / ↓ ${(s.receivedBytes / 1024).toFixed(1)} KB`;
  const ctx = s.audio.context;
  renderMicrophoneStatus(s, elapsed);
  $("client-stats").textContent = [
    `AudioContext ${ctx?.sampleRate ?? "—"} Hz / ${ctx?.state ?? "—"}`,
    `First text ${s.firstText === null ? "—" : `${Math.round(s.firstText)} ms (since ready)`}`,
    `Rebuffers ${s.audio.clock.underruns}`,
    `Missing frames ${s.audio.clock.missing}`,
    `Output base latency ${ctx ? Math.round((ctx.baseLatency || 0) * 1000) : "—"} ms`,
    `WS pending ${s.ws?.bufferedAmount ?? 0} B`,
  ].join(" · ");
  for (const [role, id] of [
    ["user", "input-level"],
    ["agent", "output-level"],
  ]) {
    const value = rms(waves[role]);
    $(id).textContent =
      value > 0 ? `${(20 * Math.log10(value)).toFixed(1)} dBFS` : "−∞ dBFS";
  }
  if (!s.closed && s.receivedFrames)
    $("output-status").textContent =
      s.receivedFrames === s.silentFrames
        ? "Model listening · No speech generated yet"
        : s.audio.clock.underruns
          ? "Playing · Rebuffered"
          : "Playing";
  drawChart(s.perf);
}

function canvasContext(id) {
  const canvas = $(id);
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (
    canvas.width !== Math.round(width * dpr) ||
    canvas.height !== Math.round(height * dpr)
  ) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function drawWave(role, color) {
  const { ctx, width, height } = canvasContext(
    role === "user" ? "user-wave" : "agent-wave",
  );
  const samples = waves[role];
  ctx.strokeStyle = "#263239";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
  ctx.strokeStyle = color;
  ctx.beginPath();
  const bars = Math.floor(width / 3);
  for (let i = 0; i < bars; i++) {
    let peak = 0;
    const a = Math.floor((i * samples.length) / bars),
      b = Math.floor(((i + 1) * samples.length) / bars);
    for (let n = a; n < b; n++) peak = Math.max(peak, Math.abs(samples[n]));
    const amp = Math.max(0.6, Math.min(height / 2 - 5, peak * height));
    ctx.moveTo(i * 3 + 1, height / 2 - amp);
    ctx.lineTo(i * 3 + 1, height / 2 + amp);
  }
  ctx.stroke();
}

function drawChart(points = []) {
  const { ctx, width, height } = canvasContext("perf-chart");
  const recent = points.slice(-60);
  const keys = [
    ["step_ms", "#a1edcf"],
    ["ear_ms", "#8ab5ee"],
    ["detok_ms", "#d5b6e7"],
  ];
  const max =
    Math.max(
      200,
      ...recent.flatMap((point) => keys.map(([key]) => point[key]?.p50 || 0)),
    ) * 1.15;
  ctx.font = "8px monospace";
  ctx.fillStyle = "#60747d";
  ctx.strokeStyle = "#29353b";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 3; i++) {
    const y = height - (i * height) / 4;
    ctx.beginPath();
    ctx.moveTo(30, y);
    ctx.lineTo(width, y);
    ctx.stroke();
    ctx.fillText(Math.round((i * max) / 4), 0, y + 3);
  }
  for (const [key, color] of keys) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    recent.forEach((point, index) => {
      const x = 32 + (index * Math.max(0, width - 35)) / 59;
      const y = height - ((point[key]?.p50 || 0) / max) * (height - 5);
      if (!index) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

$("start").addEventListener("click", () => {
  if (last?.closed) {
    last.audio.close().catch(() => {});
  }
  start();
});
$("stop").addEventListener("click", stop);
$("refresh").addEventListener("click", refresh);
$("refresh-microphones").addEventListener("click", refreshMicrophones);
navigator.mediaDevices?.addEventListener("devicechange", refreshMicrophones);
$("mute").addEventListener("change", () =>
  (active || last)?.audio.mute($("mute").checked),
);
$("resume-audio").addEventListener("click", () =>
  (active || last)?.audio.context
    ?.resume()
    .catch((error) => notice(error.message)),
);
$("download-wav").addEventListener("click", () => {
  try {
    download(
      new Blob([last.recording.wav()], { type: "audio/wav" }),
      `awaken-${last.wallTime.replace(/[:.]/g, "-")}.wav`,
    );
  } catch (error) {
    notice(error.message);
  }
});
$("download-json").addEventListener("click", () => {
  const s = last;
  const report = {
    version: 1,
    started_at: s.wallTime,
    status: s.status,
    mode: s.options.mode,
    input_file: s.options.file?.name,
    sample_rate: SAMPLE_RATE,
    frame_samples: FRAME_SAMPLES,
    recording_alignment:
      "input samples / model frame index; user left, AI right; not playback wall clock",
    start: s.options.start,
    ready: s.ready,
    end: s.end,
    health: s.health,
    microphone: s.micSettings,
    metrics: {
      sent_samples: s.sentSamples,
      input_samples: s.recording.userSamples,
      drain_context_samples: s.drainContextSamples,
      signal_samples: s.signalSamples,
      input_peak_max: s.inputPeakMax,
      received_frames: s.receivedFrames,
      silent_frames: s.silentFrames,
      first_audio_since_ready_ms: s.firstAudio,
      first_text_since_ready_ms: s.firstText,
      frame_latency_recent_ms: s.latencies,
      playback_underruns: s.audio.clock.underruns,
      missing_frames: s.audio.clock.missing,
      sent_bytes: s.sentBytes,
      received_bytes: s.receivedBytes,
    },
    utterances: s.utterances,
    frames: s.frames,
    performance: s.perf,
    events: s.events,
    dropped_events: s.droppedEvents,
  };
  download(
    new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
    `awaken-debug-${s.wallTime.replace(/[:.]/g, "-")}.json`,
  );
});
for (const input of document.querySelectorAll("input[name=mode]"))
  input.addEventListener("change", () => {
    const file = input.checked && input.value === "file";
    $("file-options").hidden = !file;
    $("mic-options").hidden = file;
  });
$("audio-file").addEventListener("change", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const file = $("audio-file").files[0];
  $("file-name").textContent = file
    ? `${file.name} · ${(file.size / 1024).toFixed(0)} KB`
    : "WAV / MP3 / browser-supported audio";
  $("file-preview").hidden = !file;
  if (file) {
    previewUrl = URL.createObjectURL(file);
    $("file-preview").src = previewUrl;
  }
});
for (const tab of document.querySelectorAll("[data-tab]"))
  tab.addEventListener("click", () => {
    for (const button of document.querySelectorAll("[data-tab]")) {
      const selected = button === tab;
      button.setAttribute("aria-selected", selected);
      $(`tab-${button.dataset.tab}`).hidden = !selected;
    }
  });
window.addEventListener("pagehide", () => {
  const s = active || last;
  if (s) {
    s.audio.stream?.getTracks().forEach((track) => track.stop());
    s.ws?.close();
  }
});
document.addEventListener("visibilitychange", () => {
  if (active && document.hidden)
    log(
      active,
      "warning",
      "Page moved to the background. The browser may pause audio or throttle streaming.",
      "warn",
    );
});
setInterval(() => {
  renderMetrics();
  drawWave("user", "#a1edcf");
  drawWave("agent", "#8ab5ee");
}, 200);
drawChart();
refresh();
refreshMicrophones();
