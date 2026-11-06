---
language:
- en
license: apache-2.0
base_model: FunAudioLLM/Fun-Audio-Chat-8B
pipeline_tag: audio-to-audio
tags:
- full-duplex
- speech-to-speech
- spoken-dialogue
- audio-language-model
---

# Awaken Full Duplex

<div align="center">

**Awaken: An Efficient Training Recipe for Full-Duplex Conversation That Retains Model Ability**

[![Blog](https://img.shields.io/badge/Blog-Project%20Page-orange)](https://github.com/avera-labs/AwakenFullDuplex/blob/main/webpage/index.html)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/avera-labs/AwakenFullDuplex)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow)](https://huggingface.co/openmartdev/awaken-full-duplex)
[![Base model](https://img.shields.io/badge/Base-Fun--Audio--Chat--8B-green)](https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B)

</div>

## Model Description

Full-duplex voice models can listen and speak at the same time, so they can acknowledge,
interrupt, and hand over turns the way people do. Teaching this usually depends on natural
conversational audio rich in backchannels. That audio is scarce, and without careful curation
it can weaken a model's reasoning and grounding. **Awaken** learns full-duplex behavior from a
small amount of such data and keeps its base model's scores.

<p align="center">
  <img width="95%" src="assets/results.png" alt="Awaken evaluation results: backchannels, groundedness, spoken logic puzzles, and spoken question answering">
</p>

### Key Features

- **Efficient**: Starts from the half-duplex [Fun-Audio-Chat-8B](https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B)
  checkpoint and teaches turn-taking with composed conversations that have explicit timing,
  overlap, and interruptions. Two supervised stages and a short reinforcement-learning stage
  are enough; there is no large-scale duplex pretraining.
- **Effective**: A targeted GRPO stage raises the backchannel pass rate from 1% (supervised
  training alone) to **63%**, while taking the floor in only **3%** of items.
- **Retains model ability**: Passes **61%** of groundedness probes, near its half-duplex base
  (66%) and well above open full-duplex models (14–35%). Solves **90%** of spoken logic
  puzzles, compared with 30–40% for open full-duplex models.

<p align="center">
  <img width="95%" src="assets/architecture.png" alt="Awaken architecture: user features, text embeddings, and audio-token embeddings enter a shared fusion layer and transformer backbone that predicts the next text token and conditions the audio decoder">
</p>

### Training Recipe

<p align="center">
  <img width="80%" src="assets/training.png" alt="Training path from Fun-Audio-Chat through streaming adaptation, two SFT stages, and backchannel-focused GRPO">
</p>

1. **Stage 1 SFT**: learns the streaming input pattern and aligned text and speech targets on
   64,000 composed conversations (windows up to 180 s).
2. **Stage 2 SFT**: blends the stage-1 weights 50/50 with the base checkpoint, then continues
   on 100,132 records with replay of the interaction data.
3. **GRPO**: improves backchannel timing from the model's own samples in conversational windows.

See the [project page](https://github.com/avera-labs/AwakenFullDuplex/blob/main/webpage/index.html) for the full data composition and training details.

## Model Details

| Attribute | Value |
|-----------|-------|
| Model Size | ~8B parameters |
| Base Model | [Fun-Audio-Chat-8B](https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B) (half-duplex) |
| Backbone | Qwen3-8B, one position per 200 ms frame |
| Input | Streaming 24 kHz user audio through a causal Whisper-style encoder (2 s attention window) |
| Output | Text tokens and 5 codec tokens per frame, rendered by the CosyVoice3 flow decoder and HiFT vocoder |
| Interaction | Full-duplex: no turns, no voice-activity detector |
| Languages | English |
| License | Apache 2.0 (same as the base model) |

## Requirements

- Linux with an NVIDIA GPU (CUDA)
- Python >= 3.10
- PyTorch >= 2.7 (tested with 2.8, CUDA 12.8)
- TensorRT 10

## Installation

```bash
git clone https://github.com/avera-labs/AwakenFullDuplex
cd AwakenFullDuplex

# 1. torch + torchaudio for YOUR CUDA driver first (the default PyPI wheel targets the newest
#    CUDA and will not load on older drivers), e.g. for CUDA 12.8:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# 2. the package
pip install -e .            # Linux + CUDA GPU; pulls transformers, tensorrt, ...
pip install -e ".[fast]"    # + flash-attn (faster decode; optional)
pip install -e ".[voice]"   # + onnxruntime, to make voice prompts from a clip
```

## Quick Start

### Download Models

```bash
pip install huggingface-hub
hf download openmartdev/awaken-full-duplex --local-dir ./pretrained_models/awaken-chat
```

The speech codec is fetched automatically from
[`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
(flow, vocoder and the estimator's ONNX export only). The TensorRT engine is built once per
machine on first use, which takes a few minutes.

### Inference

Live server (one conversation at a time, WebSocket protocol in `awaken_chat/server/protocol.py`):

```bash
awaken-server --model ./pretrained_models/awaken-chat --port 8998 [--ssl-dir certs] [--voices-dir assets/voices]
```

Offline, with a recorded user track:

```bash
awaken-offline --model ./pretrained_models/awaken-chat --input user.wav --out out/   # stereo.wav, transcript.txt, tokens.pt
```

Make a voice prompt from a reference clip (3-10 s of one speaker you have the rights to):

```bash
awaken-voice --clip speaker.wav --out assets/voices/speaker.pt
```

From Python:

```python
from awaken_chat import load_model, Decoders, DuplexSession, system_prompt

loaded = load_model("./pretrained_models/awaken-chat")   # HF repo id or local directory
decoders = Decoders(loaded.model)                         # static caches + CUDA graphs, once per process
session = DuplexSession(loaded.model, decoders, loaded.prompt_ids(system_prompt("You are a tour guide.")))
for features in ear_frames:                               # (5, 4096) per 200 ms, see models/ear.py
    out = session.step(features)                          # out.text_id, out.audio_ids, out.state
```

## Evaluation

We measure listener behavior, following the conversation, and knowledge and reasoning side by
side. Awaken and Fun-Audio-Chat-8B are run by us; other rows come from the public
InteractionBench results and the Fun-Audio-Chat technical report.

| System | Backchannel pass ↑ | Floor-take ↓ | Groundedness probes ↑ | Logic puzzles ↑ | Llama Q. ↑ | TriviaQA ↑ |
|--------|-------------------:|-------------:|----------------------:|----------------:|-----------:|-----------:|
| GPT-Realtime | 10% | 5% | 67% | 100% | – | – |
| Moshi | 78% | 22% | 14% | 30% | 65.67 | 18.46 |
| Freeze-Omni | 0% | 39% | 35% | 40% | 74.00 | 30.08 |
| PersonaPlex | 83% | 11% | 30% | 30% | – | – |
| Fun-Audio-Chat-Duplex-8B | – | – | – | – | 72.33 | 29.59 |
| Fun-Audio-Chat-8B (half-duplex base) | n/a | n/a | 66% | 100% | 76.2 | 48.7 |
| Awaken SFT | 1% | 6% | 62% | 87% | 70.7 | 35.5 |
| **Awaken GRPO** | **63%** | **3%** | **61%** | **90%** | **73.4** | **36.2** |

- **Backchannels**: 30 InteractionBench storytelling items × 3 seeds. An item passes when the
  assistant backchannels at least once and never takes the floor. A half-duplex model cannot
  speak while the user is speaking, so the base model is n/a.
- **Groundedness**: 10 conversations of 22–24 turns, then 52 probes (recall, updates, unknown
  information, constraints, false premises, nonsense questions). Our rows are means over 5–30
  runs judged on the model's text stream; published rows judge transcripts of the audio.
- **Logic puzzles**: 10 InteractionBench puzzles × 3 seeds, judged on the model's text stream.
- **Spoken QA**: our rows are means over 3 seeds; Freeze-Omni, Moshi and Fun-Audio-Chat-Duplex-8B
  are reported scores from Table 7 of the Fun-Audio-Chat technical report.

| System | OpenBookQA ↑ |
|--------|-------------:|
| Fun-Audio-Chat-8B (half-duplex) | 83.52 |
| Fun-Audio-Chat-Duplex-8B (reported) | – |
| Freeze-Omni (reported) | 30.98 |
| Moshi (reported) | 25.93 |
| Awaken SFT | 76.2 |
| **Awaken GRPO** | **75.6** |

Because our runs and the published results are scored differently, compare across systems with
care. Full methodology is on the [project page](https://github.com/avera-labs/AwakenFullDuplex/blob/main/webpage/index.html).

## Inference Details

<details>
<summary>How the inference stack works</summary>

Every 200 ms ("one frame") the model consumes

* the user's audio of that frame, through a causal **ear** (a Whisper-style encoder with a
  2 s attention window),
* its own previous frame: one **text token** and five **codec tokens** of its own voice,

and predicts the next text token and the next five codec tokens. The text lane carries the
words and two control tokens: `pad` means "listening" and, during speech, "I am done" (the
release); `sil` means "words done, audio still finishing". The codec tokens are rendered to
speech by CosyVoice3's flow-matching decoder and HiFT vocoder.

```
mic 24 kHz ─► StreamingEar ─► 5 feature vectors per frame ─┐
                                                           ▼
   own text token ──► Qwen3-8B backbone (one position per frame) ──► next text token
   own codec tokens ─┘                 │
                                       ▼
                          depth decoder (5 codec tokens / frame) ──► CosyVoice3 flow + HiFT ──► speaker
```

Both transformer stacks run through a fused, CUDA-graphed single-token decoder with static
key/value caches (`awaken_chat/inference/static_decoder.py`), so a frame costs a few
milliseconds of launch overhead instead of thousands of small kernels.

</details>

<details>
<summary>Decode parameters</summary>

`DecodeParams` (`awaken_chat/inference/params.py`) has two presets. `bench` is the recipe the
reported numbers were produced with (text T 0.7 / top-k 25, audio T 0.8 / top-k 250).
`live` is sharper inside a text burst (T 0.2 / top-k 5), penalizes repeated content tokens and
caps consecutive near-silent codec tokens; it is what the server uses by default. Any field
can be overridden per conversation in the `start` message.

</details>

<details>
<summary>Repository layout</summary>

```
awaken_chat/
  frame.py              frame rate, group size, sample rates, special token ids, system prompt
  models/               configuration, ear, token embedder, depth decoder, AwakenModel, loaders
  inference/            static decoder, session state machine, sampling, decode params, offline runner
  codec/                CosyVoice3 flow + HiFT (vendored, trimmed), streaming detokenizer, worker process, voices
  server/               WebSocket protocol, live session runner, FastAPI app
  cli.py                awaken-server / awaken-offline / awaken-voice
scripts/                ws_client.py, a paced WebSocket client that drives the server from a wav file
tests/                  CPU unit tests on tiny random models
hf_model_card/          Hugging Face model card (README + figures)
webpage/                project page
```

</details>

## License

The code and model weights are licensed under the [Apache 2.0 License](https://github.com/avera-labs/AwakenFullDuplex/blob/main/LICENSE), the same license
as the base model [Fun-Audio-Chat-8B](https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B).

## Acknowledgments

The model architecture is derived from [Fun-Audio-Chat](https://github.com/FunAudioLLM/Fun-Audio-Chat)
and the speech codec from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) (both Apache-2.0);
the text backbone is [Qwen3](https://github.com/QwenLM/Qwen3), and the ear follows
[Whisper](https://github.com/openai/whisper)'s encoder. See [`NOTICE`](https://github.com/avera-labs/AwakenFullDuplex/blob/main/NOTICE).

## Contact

- 🐛 Submit an [Issue](https://github.com/avera-labs/AwakenFullDuplex/issues)
- 💡 Submit a [Pull Request](https://github.com/avera-labs/AwakenFullDuplex/pulls)
