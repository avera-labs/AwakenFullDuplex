// Pure protocol/recording helpers. Shared by the browser and Node's test runner.
export const SAMPLE_RATE = 24000;
export const FRAME_SAMPLES = 4800;
export const CHUNK_SAMPLES = 2400;

export function encodeMic(samples) {
  const packet = new ArrayBuffer(1 + samples.length * 2);
  const view = new DataView(packet);
  view.setUint8(0, 1);
  for (let i = 0; i < samples.length; i++) {
    view.setInt16(
      1 + i * 2,
      Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767),
      true,
    );
  }
  return packet;
}

export function decodeAudio(packet) {
  const view = new DataView(packet);
  if (view.byteLength < 5) throw new Error("Truncated audio header");
  const kind = view.getUint8(0);
  if (kind !== 2 && kind !== 3)
    throw new Error(`Unknown audio packet: ${kind}`);
  if (view.byteLength !== (kind === 2 ? 5 + FRAME_SAMPLES * 2 : 5)) {
    throw new Error("Invalid 200 ms audio frame length");
  }
  const samples = new Float32Array(FRAME_SAMPLES);
  if (kind === 2) {
    for (let i = 0; i < samples.length; i++)
      samples[i] = view.getInt16(5 + i * 2, true) / 32768;
  }
  return { frame: view.getUint32(1, true), samples, silent: kind === 3 };
}

export function rms(samples) {
  let sum = 0;
  for (const value of samples) sum += value * value;
  return samples.length ? Math.sqrt(sum / samples.length) : 0;
}

export class Recording {
  constructor(enabled = true) {
    this.enabled = enabled;
    this.user = [];
    this.agent = new Map();
    this.userSamples = 0;
    this.agentEnd = 0;
  }

  addUser(samples) {
    if (this.enabled)
      this.user.push({ offset: this.userSamples, samples: samples.slice() });
    this.userSamples += samples.length;
  }

  addAgent(frame, samples, silent = false) {
    if (this.enabled && !silent) this.agent.set(frame, samples);
    this.agentEnd = Math.max(this.agentEnd, (frame + 1) * FRAME_SAMPLES);
  }

  get length() {
    return Math.max(this.userSamples, this.agentEnd);
  }

  wav() {
    if (!this.enabled || !this.length)
      throw new Error("No recording available");
    const bytes = this.length * 4;
    if (bytes > 0xffffffff - 36)
      throw new Error("Recording exceeds the WAV size limit");
    const result = new ArrayBuffer(44 + bytes);
    const view = new DataView(result);
    const ascii = (offset, value) =>
      [...value].forEach((char, i) =>
        view.setUint8(offset + i, char.charCodeAt(0)),
      );
    ascii(0, "RIFF");
    view.setUint32(4, 36 + bytes, true);
    ascii(8, "WAVE");
    ascii(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 2, true);
    view.setUint32(24, SAMPLE_RATE, true);
    view.setUint32(28, SAMPLE_RATE * 4, true);
    view.setUint16(32, 4, true);
    view.setUint16(34, 16, true);
    ascii(36, "data");
    view.setUint32(40, bytes, true);
    const write = (offset, samples, channel) => {
      for (let i = 0; i < samples.length && offset + i < this.length; i++) {
        view.setInt16(
          44 + (offset + i) * 4 + channel * 2,
          Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767),
          true,
        );
      }
    };
    for (const chunk of this.user) write(chunk.offset, chunk.samples, 0);
    for (const [frame, samples] of this.agent)
      write(frame * FRAME_SAMPLES, samples, 1);
    return result;
  }
}

// Schedule against frame indices, including silent frames. A late frame rebases the
// clock instead of overlapping speech or silently dropping a model response.
export class PlaybackClock {
  constructor(bufferSeconds = 0.2) {
    this.bufferSeconds = bufferSeconds;
    this.base = null;
    this.end = 0;
    this.lastFrame = -1;
    this.underruns = 0;
    this.missing = 0;
  }

  schedule(frame, now) {
    if (frame <= this.lastFrame)
      throw new Error(`Out-of-order audio frame ${frame}`);
    this.missing += frame - this.lastFrame - 1;
    if (this.base === null) this.base = now + this.bufferSeconds - frame * 0.2;
    let at = this.base + frame * 0.2;
    if (at < now + 0.01) {
      this.underruns++;
      this.base += now + this.bufferSeconds - at;
      at = this.base + frame * 0.2;
    }
    this.end = at + 0.2;
    this.lastFrame = frame;
    return at;
  }
}
