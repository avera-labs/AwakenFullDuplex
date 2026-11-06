import { SAMPLE_RATE, PlaybackClock } from "./core.js";

export class AudioIO {
  constructor(bufferSeconds, onPcm, onState) {
    this.clock = new PlaybackClock(bufferSeconds);
    this.onPcm = onPcm;
    this.onState = onState;
    this.sources = new Set();
    this.stream = null;
    this.closed = false;
  }

  async open() {
    if (!window.isSecureContext)
      throw new Error(
        "Audio requires HTTPS or localhost. Use SSH port forwarding to open localhost.",
      );
    this.context = new AudioContext({
      sampleRate: SAMPLE_RATE,
      latencyHint: "interactive",
    });
    if (this.context.sampleRate !== SAMPLE_RATE)
      throw new Error(
        "This browser does not support a 24 kHz AudioContext. Use a recent Chrome or Edge version.",
      );
    this.output = this.context.createGain();
    this.output.connect(this.context.destination);
    this.context.onstatechange = () => this.onState?.(this.context.state);
    await this.context.resume();
  }

  async microphone(echoCancellation, deviceId = "", autoGainControl = true) {
    if (!navigator.mediaDevices?.getUserMedia)
      throw new Error(
        "This browser cannot access the microphone. Check HTTPS or localhost access and microphone permissions.",
      );
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation,
        noiseSuppression: false,
        autoGainControl,
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      },
    });
    if (this.closed) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    this.stream = stream;
    await this.context.audioWorklet.addModule("/capture-worklet.js");
    if (this.closed) return;
    this.input = this.context.createMediaStreamSource(stream);
    this.capture = new AudioWorkletNode(this.context, "pcm-capture", {
      outputChannelCount: [1],
    });
    this.capture.port.onmessage = ({ data }) => {
      if (data.samples.length) this.onPcm(data.samples);
      if (data.type === "stopped") this.captureStopped?.();
    };
    this.input.connect(this.capture);
    // The processor outputs zeros; keeping an output connected makes it run in all browsers.
    this.capture.connect(this.context.destination);
    return stream.getAudioTracks()[0].getSettings();
  }

  startMicrophone() {
    this.capture?.port.postMessage("start");
  }

  async stopMicrophone() {
    if (!this.capture) return;
    await new Promise((resolve) => {
      const timer = setTimeout(resolve, 500);
      this.captureStopped = () => {
        clearTimeout(timer);
        resolve();
      };
      this.capture.port.postMessage("stop");
    });
    this.stream?.getTracks().forEach((track) => track.stop());
  }

  async decodeFile(file) {
    if (file.size > 100 * 1024 * 1024)
      throw new Error("Choose an audio file smaller than 100 MB.");
    const buffer = await this.context.decodeAudioData(await file.arrayBuffer());
    if (buffer.duration > 300)
      throw new Error(
        "Audio files must be no longer than 300 seconds. The default server session limit is 300 seconds.",
      );
    const mono = new Float32Array(buffer.length);
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      const channel = buffer.getChannelData(c);
      for (let i = 0; i < mono.length; i++)
        mono[i] += channel[i] / buffer.numberOfChannels;
    }
    return mono;
  }

  play({ frame, samples, silent }) {
    const at = this.clock.schedule(frame, this.context.currentTime);
    if (!silent) this.playSamples(samples, at);
  }

  playSamples(samples, at = this.context.currentTime) {
    const buffer = this.context.createBuffer(1, samples.length, SAMPLE_RATE);
    buffer.copyToChannel(samples, 0);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.output);
    this.sources.add(source);
    source.onended = () => {
      source.disconnect();
      this.sources.delete(source);
      if (!this.sources.size) this.playbackEnded?.();
    };
    source.start(at);
  }

  mute(value) {
    if (this.output) this.output.gain.value = value ? 0 : 1;
  }
  get queueMs() {
    if (!this.context || this.context.state === "closed") return 0;
    return Math.max(0, (this.clock.end - this.context.currentTime) * 1000);
  }

  async finishPlayback() {
    if (this.closed) return;
    if (this.sources.size)
      await new Promise((resolve) => {
        this.playbackEnded = resolve;
      });
    this.playbackEnded = null;
    await this.close();
  }

  async close() {
    this.closed = true;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.capture?.disconnect();
    this.input?.disconnect();
    for (const source of this.sources) {
      try {
        source.stop();
      } catch {
        /* already ended */
      }
    }
    this.sources.clear();
    this.playbackEnded?.();
    if (this.context && this.context.state !== "closed")
      await this.context.close();
  }
}
