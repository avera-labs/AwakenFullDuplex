// The AudioContext resamples the physical microphone to 24 kHz before this node.
// Capture runs on the audio thread and is never connected to audible output.
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunk = new Float32Array(2400);
    this.offset = 0;
    this.active = false;
    this.port.onmessage = ({ data }) => {
      if (data === "start") {
        this.offset = 0;
        this.active = true;
      }
      if (data === "stop") {
        this.active = false;
        const tail = this.chunk.slice(0, this.offset);
        this.offset = 0;
        this.port.postMessage({ type: "stopped", samples: tail }, [
          tail.buffer,
        ]);
      }
    };
  }

  process(inputs, outputs) {
    if (!this.active) return true;
    const channels = inputs[0];
    const count = channels[0]?.length ?? outputs[0][0].length;
    for (let i = 0; i < count; i++) {
      let sample = 0;
      for (const channel of channels) sample += channel[i] / channels.length;
      this.chunk[this.offset++] = sample;
      if (this.offset === this.chunk.length) {
        this.port.postMessage({ type: "pcm", samples: this.chunk }, [
          this.chunk.buffer,
        ]);
        this.chunk = new Float32Array(2400);
        this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
