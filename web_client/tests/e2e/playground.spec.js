import { expect, test } from "@playwright/test";
import { readFile } from "node:fs/promises";

// Generate a short stereo file at a different sample rate to exercise the browser's
// native decoder/resampler and channel downmix. No fixture downloads or user audio.
function fixture() {
  const rate = 44100,
    samples = Math.round(rate * 1.2);
  const wav = Buffer.alloc(44 + samples * 4);
  wav.write("RIFF");
  wav.writeUInt32LE(wav.length - 8, 4);
  wav.write("WAVEfmt ", 8);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(2, 22);
  wav.writeUInt32LE(rate, 24);
  wav.writeUInt32LE(rate * 4, 28);
  wav.writeUInt16LE(4, 32);
  wav.writeUInt16LE(16, 34);
  wav.write("data", 36);
  wav.writeUInt32LE(samples * 4, 40);
  for (let i = 0; i < samples; i++) {
    wav.writeInt16LE(
      Math.round(10000 * Math.sin((2 * Math.PI * 440 * i) / rate)),
      44 + i * 4,
    );
    wav.writeInt16LE(
      Math.round(4000 * Math.sin((2 * Math.PI * 660 * i) / rate)),
      46 + i * 4,
    );
  }
  return { name: "user-44k-stereo.wav", mimeType: "audio/wav", buffer: wav };
}

async function fileMode(page) {
  await page.getByText("↥ Audio file", { exact: true }).click();
  await page.locator("#audio-file").setInputFiles(fixture());
  await page.locator("#tail").fill("0.4");
}

async function saveDownload(page, id) {
  const promise = page.waitForEvent("download");
  await page.locator(id).click();
  const download = await promise;
  return readFile(await download.path());
}

async function report(page) {
  return JSON.parse((await saveDownload(page, "#download-json")).toString());
}

async function complete(page) {
  await expect(page.locator("#connection")).toHaveText("Session ended");
  await expect(page.locator("#download-json")).toBeEnabled();
}

test.beforeEach(async ({ page, request }) => {
  await request.post("/_test/scenario", { data: {} });
  await page.goto("/");
  await expect(page.locator("#health-status")).toHaveText(
    "Inference server ready",
  );
});

test("static audio runs through the proxy, resamples, exports stereo WAV and complete debug data", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => {
    window.micCalls = 0;
    const original = navigator.mediaDevices.getUserMedia.bind(
      navigator.mediaDevices,
    );
    navigator.mediaDevices.getUserMedia = (...args) => {
      window.micCalls++;
      return original(...args);
    };
  });
  await page.reload();
  await fileMode(page);
  await page.locator("summary").click();
  await page.locator("#seed").fill("73");
  await page.locator("#decode").fill('{"text_temperature":0.2}');
  await page.locator("#start").click();
  await complete(page);
  expect(await page.evaluate(() => window.micCalls)).toBe(0);
  await expect(page.locator("#transcript")).toContainText("protocol simulator");
  await expect(page.locator("#step-p50")).toHaveText("12 ms");
  const debug = await report(page);
  expect(debug.status).toBe("completed");
  expect(debug.ready.seed).toBe(73);
  expect(debug.ready.decode.text_temperature).toBe(0.2);
  expect(debug.metrics.input_samples).toBe(38400);
  expect(debug.metrics.sent_samples).toBe(40800);
  expect(debug.metrics.received_frames).toBe(8);
  expect(debug.metrics.missing_frames).toBe(0);
  expect(debug.frames).toHaveLength(8);
  const wav = await saveDownload(page, "#download-wav");
  expect(wav.toString("ascii", 0, 4)).toBe("RIFF");
  expect(wav.readUInt16LE(22)).toBe(2);
  expect(wav.readUInt32LE(24)).toBe(24000);
  expect(wav.readUInt32LE(40)).toBe(38400 * 4);
  const left = (i) => wav.readInt16LE(44 + i * 4);
  const right = (i) => wav.readInt16LE(46 + i * 4);
  expect(
    Array.from({ length: 28800 }, (_, i) => Math.abs(left(i))).some(
      (value) => value > 1000,
    ),
  ).toBe(true);
  expect(
    Array.from({ length: 9600 }, (_, i) => left(28800 + i)).every(
      (value) => value === 0,
    ),
  ).toBe(true);
  expect(
    Array.from({ length: 9600 }, (_, i) => right(i)).every(
      (value) => value === 0,
    ),
  ).toBe(true);
  expect(
    Array.from({ length: 4800 }, (_, i) => Math.abs(right(9600 + i))).some(
      (value) => value > 1000,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
});

test("streaming ear lookahead releases the final file frame without a drain timeout", async ({
  page,
  request,
}) => {
  await request.post("/_test/scenario", { data: { holdback_samples: 2400 } });
  await fileMode(page);
  await page.locator("#start").click();
  await expect(page.locator("#connection")).toHaveText("Session ended", {
    timeout: 5000,
  });
  const debug = await report(page);
  expect(debug.metrics.received_frames).toBe(8);
  expect(debug.metrics.input_samples).toBe(38400);
  expect(debug.metrics.drain_context_samples).toBe(2400);
  expect(debug.metrics.sent_samples).toBe(40800);
  expect(debug.end.frames).toBe(8);
  expect(debug.events.filter((event) => event.type === "warning")).toEqual([]);
  await expect(page.locator("#notice")).toBeHidden();
  const wav = await saveDownload(page, "#download-wav");
  expect(wav.readUInt32LE(40)).toBe(38400 * 4);
});

test("virtual microphone sends continuously while model audio and text arrive; stop drains and releases audio", async ({
  page,
  request,
}) => {
  await request.post("/_test/scenario", { data: { holdback_samples: 2400 } });
  await page.locator("#start").click();
  await expect(page.locator("#transcript")).toContainText("protocol simulator");
  await expect(page.locator("#connection")).toHaveText("Conversation active");
  await expect(page.locator("#recording-status")).toContainText("Recording");
  await expect(page.locator("#input-duration")).not.toHaveText("0.0 s sent");
  await expect(page.locator("#output-duration")).not.toHaveText(
    "0 frames · 0 silent",
  );
  await page.locator("#mute").check();
  await page.locator("#stop").click();
  await complete(page);
  const debug = await report(page);
  expect(debug.mode).toBe("mic");
  expect(debug.microphone).toBeTruthy();
  expect(debug.metrics.sent_samples).toBeGreaterThan(24000);
  expect(debug.metrics.received_frames * 4800).toBe(
    debug.metrics.input_samples,
  );
  expect(debug.metrics.sent_samples - debug.metrics.input_samples).toBe(2400);
  expect(debug.events.filter((event) => event.type === "warning")).toEqual([]);
  await expect(page.locator("#client-stats")).toContainText("/ closed");
  await expect(page.locator("#download-wav")).toBeEnabled();
});

test("recording opt-out retains debug data and disables WAV download", async ({
  page,
}) => {
  await fileMode(page);
  await page.locator("#record").uncheck();
  await page.locator("#start").click();
  await complete(page);
  await expect(page.locator("#download-wav")).toBeDisabled();
  await expect(page.locator("#download-json")).toBeEnabled();
});

test("ending a session preserves queued speech while the audio context is suspended", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const NativeContext = window.AudioContext;
    window.AudioContext = class extends NativeContext {
      constructor(options) {
        super(options);
        window.playbackContext = this;
      }
    };
  });
  await page.reload();
  await fileMode(page);
  await page.locator("#start").click();
  await expect(page.locator("#connection")).toHaveText("Conversation active");
  await page.evaluate(() => window.playbackContext.suspend());
  await complete(page);
  // Wall-clock time advances while the Web Audio clock and its queued speech are paused.
  await page.waitForTimeout(2500);
  expect(await page.evaluate(() => window.playbackContext.state)).toBe(
    "suspended",
  );
  await expect(page.locator("#resume-audio")).toBeVisible();
  await page.locator("#resume-audio").click();
  await expect(page.locator("#output-status")).toHaveText("Playback finished", {
    timeout: 5000,
  });
  expect(await page.evaluate(() => window.playbackContext.state)).toBe(
    "closed",
  );
});

test("selected microphone and automatic gain reach getUserMedia", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const original = navigator.mediaDevices.getUserMedia.bind(
      navigator.mediaDevices,
    );
    navigator.mediaDevices.getUserMedia = (constraints) => {
      window.requestedAudio = constraints.audio;
      return original(constraints);
    };
  });
  await page.reload();
  const devices = await page.evaluate(async () =>
    (await navigator.mediaDevices.enumerateDevices()).filter(
      (device) => device.kind === "audioinput" && device.deviceId,
    ),
  );
  expect(devices.length).toBeGreaterThan(0);
  await page.locator("#microphone-device").selectOption(devices[0].deviceId);
  await page.locator("#start").click();
  await expect(page.locator("#connection")).toHaveText("Conversation active");
  const audio = await page.evaluate(() => window.requestedAudio);
  expect(audio.deviceId).toEqual({ exact: devices[0].deviceId });
  expect(audio.autoGainControl).toBe(true);
  await page.locator("#stop").click();
  await complete(page);
});

test("silent microphone shows an actionable message instead of pretending audio is playing", async ({
  page,
}) => {
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => {
      const context = new AudioContext({ sampleRate: 24000 });
      const source = new ConstantSourceNode(context, { offset: 0 });
      const destination = context.createMediaStreamDestination();
      source.connect(destination);
      source.start();
      await context.resume();
      destination.stream
        .getTracks()[0]
        .addEventListener("ended", () => context.close());
      return destination.stream;
    };
  });
  await page.reload();
  await page.locator("#start").click();
  await expect(page.locator("#microphone-status")).toContainText(
    "No sound detected",
    { timeout: 8000 },
  );
  await page.locator("#stop").click();
  await complete(page);
  const debug = await report(page);
  expect(debug.metrics.signal_samples).toBe(0);
  expect(debug.metrics.input_peak_max).toBe(0);
});

test("busy inference server displays its error and releases the browser audio resources", async ({
  page,
  request,
}) => {
  await request.post("/_test/scenario", { data: { live: true } });
  await fileMode(page);
  await page.locator("#start").click();
  await expect(page.locator("#notice")).toContainText(
    "another conversation is running",
  );
  await expect(page.locator("#start")).toBeVisible();
  await expect(page.locator("#client-stats")).toContainText("/ closed");
  expect((await report(page)).status).toBe("error");
});

test("mid-stream inference failure preserves partial recording and allows a new session", async ({
  page,
  request,
}) => {
  await request.post("/_test/scenario", { data: { fail_after: 3 } });
  await fileMode(page);
  await page.locator("#start").click();
  await expect(page.locator("#notice")).toContainText(
    "simulated inference failure",
  );
  await expect(page.locator("#download-wav")).toBeEnabled();
  expect((await report(page)).status).toBe("error");
  await request.post("/_test/scenario", { data: {} });
  await page.locator("#start").click();
  await complete(page);
});

test("upstream handshake failure is actionable and does not leave the UI running", async ({
  page,
  request,
}) => {
  await request.post("/_test/scenario", { data: { offline: true } });
  await fileMode(page);
  await page.locator("#start").click();
  await expect(page.locator("#notice")).toContainText(
    "Inference connection failed",
  );
  await expect(page.locator("#start")).toBeVisible();
  await expect(page.locator("#health-status")).toHaveText(
    "Inference server unreachable",
  );
});

test("pending microphone permission can be cancelled before a connection is opened", async ({
  page,
}) => {
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = () => new Promise(() => {});
  });
  await page.reload();
  await page.locator("#start").click();
  await expect(page.locator("#connection")).toHaveText("Preparing audio");
  await page.locator("#stop").click();
  await expect(page.locator("#connection")).toHaveText("Session cancelled");
  await expect(page.locator("#client-stats")).toContainText("/ closed");
  const debug = await report(page);
  expect(debug.metrics.sent_samples).toBe(0);
  expect(debug.ready).toBeUndefined();
});

test("mobile layout remains usable without horizontal overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await fileMode(page);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.locator("#start").click();
  await complete(page);
  await expect(page.locator("#download-wav")).toBeEnabled();
});
