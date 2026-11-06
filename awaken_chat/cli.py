"""Command-line entry points: ``awaken-server``, ``awaken-offline``, ``awaken-voice``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from awaken_chat.codec.detokenizer import CODEC_REPO


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def server_main() -> None:
    import uvicorn

    from awaken_chat.server.app import ServerSettings, create_app

    ap = argparse.ArgumentParser(
        description="Awaken Chat live server (WebSocket, one conversation at a time)")
    ap.add_argument("--model", required=True, help="checkpoint directory or Hub repo id")
    ap.add_argument("--codec", default=CODEC_REPO, help="CosyVoice3 directory or Hub repo id")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8998)
    ap.add_argument("--ssl-dir", default="", help="directory with key.pem and cert.pem")
    ap.add_argument("--voices-dir", default=None, help="directory of *.pt voice prompts")
    ap.add_argument("--default-voice", default="")
    ap.add_argument("--no-detok-fp16", action="store_true")
    args = ap.parse_args()
    _logging()
    overrides = {k: v for k, v in {"voices_dir": args.voices_dir}.items() if v is not None}
    settings = ServerSettings(model=args.model, codec=args.codec, host=args.host, port=args.port,
                              ssl_dir=args.ssl_dir, default_voice=args.default_voice,
                              detok_fp16=not args.no_detok_fp16, **overrides)
    kw = {}
    if settings.ssl_dir:
        d = Path(settings.ssl_dir)
        kw = {"ssl_keyfile": str(d / "key.pem"), "ssl_certfile": str(d / "cert.pem")}
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info", **kw)


def offline_main() -> None:
    import torch

    from awaken_chat.codec.detokenizer import AudioDetokenizer, VoicePrompt
    from awaken_chat.inference.offline import load_user_track, run_offline
    from awaken_chat.inference.params import PRESETS
    from awaken_chat.inference.session import Decoders
    from awaken_chat.models.loaders import load_model

    ap = argparse.ArgumentParser(description="Run a recorded user track through the model offline")
    ap.add_argument("--model", required=True)
    ap.add_argument("--codec", default=CODEC_REPO)
    ap.add_argument("--input", required=True, help="user audio file (any rate, mono or stereo)")
    ap.add_argument("--out", required=True, help="output directory (stereo.wav, transcript.txt, tokens.pt)")
    ap.add_argument("--prompt", default="", help="extra system-prompt sentence (persona, instructions)")
    ap.add_argument("--voice", default="", help="voice prompt .pt (default: unconditioned)")
    ap.add_argument("--preset", default="bench", choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tail-s", type=float, default=5.0, help="silence appended after the input")
    ap.add_argument("--no-audio", action="store_true", help="skip rendering speech")
    args = ap.parse_args()
    _logging()
    loaded = load_model(args.model)
    decoders = Decoders(loaded.model)
    detok = None if args.no_audio else AudioDetokenizer.load(args.codec, prefer_trt=False, warmup=False)
    voice = VoicePrompt.load(args.voice) if args.voice else None
    result = run_offline(loaded, decoders, load_user_track(args.input), PRESETS[args.preset],
                         args.prompt, args.seed, args.tail_s, detok, voice)
    result.save(args.out)
    for u in result.utterances:
        print(f"[{u.start_s:6.1f}-{u.end_s:6.1f}] {u.text}")
    print(f"{len(result.frames)} frames -> {args.out}")
    del decoders
    torch.cuda.empty_cache()


def voice_main() -> None:
    from awaken_chat.codec.voice import VoiceExtractor, trim_prompt

    ap = argparse.ArgumentParser(description="Make a voice prompt (.pt) from a reference clip")
    ap.add_argument("--clip", required=True, help="reference speech, 3-10 s, one speaker")
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--codec", default=CODEC_REPO)
    ap.add_argument("--max-s", type=float, default=6.0, help="seconds of the clip to use")
    ap.add_argument("--keep-s", type=float, default=3.0,
                    help="seconds kept in the prompt (shorter = cheaper per frame)")
    args = ap.parse_args()
    _logging()
    voice = VoiceExtractor(args.codec).extract(args.clip, max_s=args.max_s)
    voice = trim_prompt(voice, args.keep_s)
    voice.save(args.out)
    print(f"voice prompt: {voice.prompt_token.shape[1]} tokens -> {args.out}")
