#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional

from asr_kotoba import find_kotoba_ggml_model, run_kotoba_whispercpp
from asr_whisper import run_whispercpp
from audioengine import AudioConfig, extract_and_clean_audio
from merge_subs import auto_merge_and_insert, progress_bar, run
from ollama_proofread import OllamaConfig
from rescue_pass import run_rescue_pass, RescueConfig
from settings import (
    ensure_project_dirs,
    resolve_whisper_model,
    DIR_ASR,
    DIR_OUT,
    DIR_POST,
    WCPP_BIN,
    KOTOBA_GGML_DIR,
    DEFAULT_WORKDIR,
    WHISPER_TAG,
    KOTOBA_TAG,
    FINAL_TAG,
    COMPARE_TAG,
    MAX_SRT_LINE_DUR,
)

# ------------------------------------------------------------
# Small basic checks + helpers
# ------------------------------------------------------------


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s.strip()))


def ensure_tools(*, whisper_model: Path) -> None:
    missing: list[str] = []
    for tool in ("ffmpeg", "ffprobe", "yt-dlp"):
        if not have(tool):
            missing.append(tool)

    if not WCPP_BIN.exists():
        missing.append(f"whisper-cli ({WCPP_BIN})")

    if not whisper_model.exists():
        missing.append(f"whisper model ({whisper_model})")

    if missing:
        raise SystemExit(
            "Missing requirements:\n" + "\n".join(f"- {m}" for m in missing)
        )


def yt_dlp_download(url: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--ignore-config",
        "-P",
        str(outdir),
        "-o",
        "%(title)s [%(id)s].%(ext)s",
        "-f",
        "bestvideo+bestaudio/best",
        "--print",
        "after_move:filepath",
        url,
    ]
    res = run(cmd, check=True)
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("yt-dlp returned no output filepath")
    path = Path(lines[-1])
    if not path.exists():
        raise RuntimeError(f"yt-dlp said it downloaded to {path}, but file not found")
    return path


def iter_media_files(folder: Path) -> Iterable[Path]:
    exts = {
        ".mkv",
        ".mp4",
        ".webm",
        ".avi",
        ".mov",
        ".m4v",
        ".mp3",
        ".flac",
        ".ogg",
        ".wav",
    }
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


# ------------------------------------------------------------
# One-file pipeline
# ------------------------------------------------------------


def _copy_text_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        dst.write_bytes(src.read_bytes())


def process_one(input_arg: str, workdir: Path, args) -> None:
    ensure_project_dirs()

    whisper_model = resolve_whisper_model()
    ensure_tools(whisper_model=whisper_model)

    t0 = time.perf_counter()

    # If it's a URL, grab the media first. Otherwise, treat it as a local path.
    if is_url(input_arg):
        media_path = yt_dlp_download(input_arg, workdir)
    else:
        media_path = Path(input_arg).expanduser()

    if not media_path.exists():
        raise SystemExit(f"Input not found: {media_path}")

    stem = media_path.stem

    # Folder layout:
    #   out/<stem>/   -> wav + json + intermediate stuff for this run
    #   asr/          -> final 'keeper' SRTs
    #   post/         -> optional debug logs
    out_dir = DIR_OUT / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[subloom] {media_path}")
    print(f"[subloom] whisper model: {whisper_model.name}")

    wav = out_dir / f"{stem}.wav"
    wav_stereo = out_dir / f"{stem}.stereo.wav"

    audio_cfg = AudioConfig(
        audio_stream=args.audio_stream,
        audio_auto=not args.no_audio_auto,
        min_audio_sec=args.min_audio_sec,
        no_clean=args.no_clean,
        preset=args.audio_preset,
        gain_db=args.audio_gain_db,
    )

    clean = extract_and_clean_audio(
        media_path, wav, audio_cfg, out_wav_stereo=wav_stereo
    )

    outbase = out_dir / stem

    whisper_srt_out = out_dir / f"{stem}.{WHISPER_TAG}.srt"
    whisper_json_out = out_dir / f"{stem}.{WHISPER_TAG}.json"
    kotoba_json_out = out_dir / f"{stem}.{KOTOBA_TAG}.json"

    whisper_srt = DIR_ASR / f"{stem}.{WHISPER_TAG}.srt"
    final_srt = DIR_ASR / f"{stem}.{FINAL_TAG}.srt"

    compare_log: Optional[Path] = None
    if args.compare_log:
        compare_log = DIR_POST / f"{stem}.{COMPARE_TAG}.txt"

    # --------------------------------------------------------
    # 1) Whisper (timing backbone)
    # --------------------------------------------------------
    whisper_ready = whisper_srt.exists() and whisper_json_out.exists()
    if args.resume and whisper_ready:
        print("[1/3] whisper.cpp (resume)")
    else:
        print("[1/3] whisper.cpp ...")
        w = run_whispercpp(
            clean,
            outbase,
            whisper_bin=WCPP_BIN,
            whisper_model=whisper_model,
            whisper_tag=WHISPER_TAG,
            max_srt_line_dur=MAX_SRT_LINE_DUR,
        )
        _copy_text_file(w.srt_path, whisper_srt_out)
        _copy_text_file(whisper_srt_out, whisper_srt)

    if not whisper_srt.exists():
        raise RuntimeError(f"Whisper SRT missing: {whisper_srt}")
    if not whisper_json_out.exists():
        raise RuntimeError(f"Whisper JSON missing: {whisper_json_out}")

    # --------------------------------------------------------
    # 2) Kotoba (secondary check / correction source)
    # --------------------------------------------------------
    if args.kotoba_ggml:
        model_path = Path(args.kotoba_ggml).expanduser()
    else:
        model_path = find_kotoba_ggml_model(KOTOBA_GGML_DIR)

    if not model_path:
        raise RuntimeError(
            f"No Kotoba GGML .bin found in {KOTOBA_GGML_DIR} (or pass --kotoba-ggml)"
        )

    if args.resume and kotoba_json_out.exists():
        print("[2/3] kotoba check (resume)")
    else:
        print("[2/3] kotoba check ...")
        run_kotoba_whispercpp(
            clean,
            outbase,
            model_path,
            whisper_bin=WCPP_BIN,
            kotoba_tag=KOTOBA_TAG,
        )

    if not kotoba_json_out.exists():
        raise RuntimeError(f"Kotoba JSON missing: {kotoba_json_out}")

    whisper_srt_for_merge = whisper_srt

    # --------------------------------------------------------
    # Pass 3: Rescue
    # --------------------------------------------------------
    preview_srt = out_dir / "rescue" / "rescue_merged_preview.srt"
    if args.no_rescue:
        pass
    elif args.resume and preview_srt.exists():
        print("[2.5/3] rescue pass (resume)")
        whisper_srt_for_merge = preview_srt
    else:
        print("[2.5/3] rescue pass (targeted re-listen) ...")

        rescue_cfg = RescueConfig(
            min_gap_s=args.rescue_min_gap_s,
            pad_s=args.rescue_pad_s,
            max_window_s=args.rescue_max_window_s,
            max_windows=args.rescue_max_windows,
        )

        rescue_audio = wav_stereo if wav_stereo.exists() else clean

        added_srt, rescue_report = run_rescue_pass(
            clean_wav=rescue_audio,
            base_srt=whisper_srt_for_merge,
            out_dir=out_dir / "rescue",
            run_whispercpp_fn=run_whispercpp,
            whisper_bin=WCPP_BIN,
            whisper_model=whisper_model,
            max_srt_line_dur=MAX_SRT_LINE_DUR,
            cfg=rescue_cfg,
        )

        if preview_srt.exists():
            whisper_srt_for_merge = preview_srt

        if rescue_report:
            print(f"[subloom] rescue report: {rescue_report}")
        if added_srt:
            print(f"[subloom] rescue added lines: {added_srt}")
            print(f"[subloom] rescue preview: {preview_srt}")
        else:
            print("[subloom] rescue found nothing new (nice)")

    # --------------------------------------------------------
    # 3) Merge + optional Ollama -> FINAL
    # --------------------------------------------------------
    print("[3/3] auto-merge + insert -> FINAL SRT ...")

    ollama_cfg = None
    if args.ollama:
        ollama_cfg = OllamaConfig(
            model=args.ollama_model,
            url=args.ollama_url,
            style=args.ollama_style,
            window_sec=args.ollama_window_sec,
            max_chars=args.ollama_max_chars,
            max_retries=args.ollama_retries,
            skip_music_lines=not args.ollama_no_skip_music,
        )

    auto_merge_and_insert(
        whisper_srt=whisper_srt_for_merge,
        kotoba_json=kotoba_json_out,
        out_final_srt=final_srt,
        out_compare_log=compare_log,
        ollama_cfg=ollama_cfg,
    )

    if not args.keep_wav:
        for p in (
            wav,
            wav_stereo,
            wav.with_suffix(".clean.wav"),
            wav.with_suffix(".bad.wav"),
        ):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    dt = time.perf_counter() - t0
    mins = int(dt // 60)
    secs = dt % 60
    print(f"[subloom] Done: {final_srt}")
    print(
        f"[subloom] Finished in {mins}m {secs:.1f}s"
        if mins
        else f"[subloom] Finished in {secs:.1f}s"
    )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="subloom",
        description="Whisper timing SRT + Kotoba check + rescue pass + optional Ollama proofreading",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--workdir", default=str(DEFAULT_WORKDIR), help="where URL downloads go"
        )
        p.add_argument(
            "--kotoba-ggml",
            default=None,
            help=f"Path to Kotoba GGML .bin (default auto from {KOTOBA_GGML_DIR})",
        )
        p.add_argument(
            "--resume",
            action="store_true",
            help="Skip whisper/kotoba/rescue if outputs already exist",
        )
        p.add_argument("--keep-wav", action="store_true", help="Keep extracted WAVs")
        p.add_argument(
            "--no-clean", action="store_true", help="Skip ffmpeg filter pass"
        )
        p.add_argument(
            "--compare-log", action="store_true", help="Write compare/debug log"
        )

        # Audio
        p.add_argument(
            "--audio-stream",
            default=None,
            help="Force audio stream mapping (example: 0:a:1). Overrides auto-pick.",
        )
        p.add_argument(
            "--no-audio-auto",
            action="store_true",
            help="Disable auto audio stream selection.",
        )
        p.add_argument(
            "--min-audio-sec",
            type=float,
            default=20.0,
            help="If extracted WAV is shorter than this, treat it as broken and retry extraction.",
        )
        p.add_argument(
            "--audio-preset",
            choices=["balanced", "strong", "anime"],
            default="balanced",
            help="Audio preprocessing preset.",
        )
        p.add_argument(
            "--audio-gain-db",
            type=float,
            default=0.0,
            help="Optional gain applied before filtering (example: 4.0 or -2.0).",
        )
        # RNNoise (required; defaults are hard-set in settings.py)
        p.add_argument(
            "--rnnoise-mode",
            choices=["fixed", "chunk"],
            default="chunk",
            help=argparse.SUPPRESS,
        )
        p.add_argument(
            "--rnnoise-mix", type=float, default=0.51, help=argparse.SUPPRESS
        )
        p.add_argument(
            "--rnnoise-chunk-s", type=float, default=20.0, help=argparse.SUPPRESS
        )
        p.add_argument(
            "--rnnoise-mix-min", type=float, default=0.28, help=argparse.SUPPRESS
        )
        p.add_argument(
            "--rnnoise-mix-max", type=float, default=0.80, help=argparse.SUPPRESS
        )
        p.add_argument(
            "--rnnoise-frame-ms", type=int, default=30, help=argparse.SUPPRESS
        )

        # Ollama
        p.add_argument(
            "--ollama",
            action="store_true",
            help="Enable Ollama proofreading on FINAL subtitles",
        )
        p.add_argument(
            "--ollama-model", default="llama3.1:latest", help="Ollama model name"
        )
        p.add_argument(
            "--ollama-style", choices=["neutral", "anime", "formal"], default="neutral"
        )
        p.add_argument("--ollama-window-sec", type=float, default=45.0)
        p.add_argument("--ollama-max-chars", type=int, default=1000)
        p.add_argument("--ollama-retries", type=int, default=1)
        p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
        p.add_argument("--ollama-no-skip-music", action="store_true")

        # Rescue
        p.add_argument(
            "--no-rescue",
            action="store_true",
            help="Disable the rescue pass (debug/speed tests)",
        )
        p.add_argument(
            "--rescue-min-gap-s",
            type=float,
            default=2.5,
            help="Minimum subtitle gap (seconds) to trigger a rescue window.",
        )
        p.add_argument(
            "--rescue-pad-s",
            type=float,
            default=0.35,
            help="Padding (seconds) added before/after each rescue window.",
        )
        p.add_argument(
            "--rescue-max-window-s",
            type=float,
            default=9.0,
            help="Max length (seconds) for a single rescue window.",
        )
        p.add_argument(
            "--rescue-max-windows",
            type=int,
            default=18,
            help="Safety cap for how many rescue windows we will try per file.",
        )

    ap_run = sub.add_parser("run", help="Process one file path or URL")
    ap_run.add_argument("input", help="file path or URL")
    add_common(ap_run)

    ap_batch = sub.add_parser(
        "batch", help="Process all media files in a folder (recursive)"
    )
    ap_batch.add_argument("folder", help="folder to scan recursively")
    add_common(ap_batch)

    args = ap.parse_args()
    ensure_project_dirs()

    workdir = Path(args.workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "run":
        process_one(args.input, workdir, args)
        return

    if args.cmd == "batch":
        folder = Path(args.folder).expanduser()
        if not folder.is_dir():
            raise SystemExit(f"Not a folder: {folder}")

        files = list(iter_media_files(folder))
        total = len(files)
        if total == 0:
            raise SystemExit(f"No media files found under: {folder}")

        for idx, f in enumerate(files, start=1):
            try:
                progress_bar("[batch]", idx - 1, total)
                process_one(str(f), workdir, args)
            except Exception as e:
                print(f"[error] {f}: {e}", file=sys.stderr)

        progress_bar("[batch]", total, total)
        sys.stderr.write("\n")
        sys.stderr.flush()
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
