import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import {
  CHUNK_SAMPLES,
  FRAME_SAMPLES,
  SAMPLE_RATE,
  PlaybackClock,
  Recording,
  decodeAudio,
  encodeMic,
} from "../static/core.js";

test("microphone wire format clips to PCM16, with a one-byte kind and little-endian samples", () => {
  const packet = new DataView(encodeMic(Float32Array.of(-2, -0.5, 0, 0.5, 2)));
  assert.equal(packet.getUint8(0), 1);
  assert.equal(packet.byteLength, 11);
  assert.equal(packet.getInt16(1, true), -32767);
  assert.equal(packet.getInt16(3, true), -16383);
  assert.equal(packet.getInt16(7, true), 16384);
  assert.equal(packet.getInt16(9, true), 32767);
});

test("audio packets preserve frame index and signed samples; silence has a full frame duration", () => {
  const packet = new ArrayBuffer(9605),
    view = new DataView(packet);
  view.setUint8(0, 2);
  view.setUint32(1, 71, true);
  view.setInt16(5, -16384, true);
  const output = decodeAudio(packet);
  assert.equal(output.frame, 71);
  assert.equal(output.samples.length, FRAME_SAMPLES);
  assert.equal(output.samples[0], -0.5);
  const silence = new ArrayBuffer(5),
    silentView = new DataView(silence);
  silentView.setUint8(0, 3);
  silentView.setUint32(1, 72, true);
  assert.equal(decodeAudio(silence).samples.length, FRAME_SAMPLES);
  assert.equal(decodeAudio(silence).silent, true);
  assert.throws(() => decodeAudio(new ArrayBuffer(4)), /Truncated/);
  assert.throws(() => decodeAudio(new ArrayBuffer(5)), /Unknown/);
  view.setUint8(0, 3);
  assert.throws(() => decodeAudio(packet), /length/);
});

test("WAV records user LEFT / AI RIGHT on the frame grid, with missing and silent frames zero-filled", () => {
  const recording = new Recording();
  recording.addUser(new Float32Array(FRAME_SAMPLES).fill(0.5));
  recording.addUser(new Float32Array(FRAME_SAMPLES).fill(-0.5));
  recording.addAgent(0, new Float32Array(FRAME_SAMPLES), true);
  recording.addAgent(2, new Float32Array(FRAME_SAMPLES).fill(0.25));
  const wav = new DataView(recording.wav());
  assert.equal(
    new TextDecoder().decode(new Uint8Array(wav.buffer, 0, 4)),
    "RIFF",
  );
  assert.equal(wav.getUint16(22, true), 2);
  assert.equal(wav.getUint32(24, true), SAMPLE_RATE);
  assert.equal(wav.getUint32(40, true), 3 * FRAME_SAMPLES * 4);
  const sample = (frame, channel) =>
    wav.getInt16(44 + frame * FRAME_SAMPLES * 4 + channel * 2, true);
  assert.equal(sample(0, 0), 16384);
  assert.equal(sample(0, 1), 0);
  assert.equal(sample(1, 0), -16383);
  assert.equal(sample(1, 1), 0);
  assert.equal(sample(2, 0), 0);
  assert.equal(sample(2, 1), 8192);
});

test("recording opt-out retains no raw samples", () => {
  const recording = new Recording(false);
  recording.addUser(new Float32Array(100));
  recording.addAgent(2, new Float32Array(FRAME_SAMPLES));
  assert.equal(recording.user.length, 0);
  assert.equal(recording.agent.size, 0);
  assert.throws(() => recording.wav(), /No recording/);
});

test("playback respects silent frame spacing, repairs underruns, and rejects duplicates", () => {
  const clock = new PlaybackClock(0.2);
  assert.equal(clock.schedule(0, 10), 10.2);
  assert.equal(clock.schedule(1, 10.15), 10.399999999999999);
  const late = clock.schedule(2, 11);
  assert.ok(Math.abs(late - 11.2) < 1e-10);
  assert.equal(clock.underruns, 1);
  clock.schedule(4, 11.3);
  assert.equal(clock.missing, 1);
  assert.throws(() => clock.schedule(4, 11.4), /Out-of-order/);
});

test("real capture worklet batches 100 ms, downmixes, flushes partial input, and stops capturing", async () => {
  let Processor;
  const messages = [];
  class Base {
    constructor() {
      this.port = { postMessage: (message) => messages.push(message) };
    }
  }
  const context = vm.createContext({
    AudioWorkletProcessor: Base,
    Float32Array,
    registerProcessor: (_, cls) => {
      Processor = cls;
    },
  });
  vm.runInContext(
    await readFile(
      new URL("../static/capture-worklet.js", import.meta.url),
      "utf8",
    ),
    context,
  );
  const capture = new Processor();
  const input = [
    [new Float32Array(128).fill(0.5), new Float32Array(128).fill(-0.25)],
  ];
  const output = [[new Float32Array(128)]];
  capture.process(input, output);
  assert.equal(messages.length, 0);
  capture.port.onmessage({ data: "start" });
  for (let i = 0; i < 20; i++) capture.process(input, output);
  assert.equal(messages.length, 1);
  assert.equal(messages[0].samples.length, CHUNK_SAMPLES);
  assert.ok(messages[0].samples.every((sample) => sample === 0.125));
  capture.port.onmessage({ data: "stop" });
  assert.equal(messages[1].type, "stopped");
  assert.equal(messages[1].samples.length, 160);
  for (let i = 0; i < 20; i++) capture.process(input, output);
  assert.equal(messages.length, 2);
});
