#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional

from asr_kotoba import find_kotoba_ggml_model, run_kotoba_whispercpp
from asr_whisper import run_whispercpp
from merge_subs import auto_merge_and_insert, progress_bar, run
from ollama_proofread import OllamaConfig

# ----------------------------
# Paths + defaults (portable / folder-local)
# ----------------------------

APP_DIR = Path(__file__).resolve().parent

# Folder routing:
# - asr/  : final usable subtitles (whisper.srt + final.srt)
# - out/  : raw junk per-title (wav/json/txt)
# - post/ : logs/debug (compare)
# - work/ : downloads/temp
# - models/: local models + whisper.cpp (optional)
DIR_ASR = APP_DIR / "asr"
DIR_OUT = APP_DIR / "out"
DIR_POST = APP_DIR / "post"
DIR_WORK = APP_DIR / "work"
DIR_MODELS = APP_DIR / "models"

# whisper.cpp stored inside subloom/models/whisper.cpp
WCPP_BASE = DIR_MODELS / "whisper.cpp"
WCPP_BIN = WCPP_BASE / "build/bin/whisper-cli"
WCPP_MODEL = WCPP_BASE / "models/ggml-large-v3.bin"

# Kotoba models stored inside subloom/models/kotoba/
KOTOBA_GGML_DIR = DIR_MODELS / "kotoba"

DEFAULT_WORKDIR = DIR_WORK

WHISPER_TAG = "whisper"
KOTOBA_TAG = "kotoba"
FINAL_TAG = "final"
COMPARE_TAG = "compare"

MAX_SRT_LINE_DUR = 10.0


def ensure_project_dirs() -> None:
    for d in (DIR_ASR, DIR_OUT, DIR_POST, DIR_WORK, DIR_MODELS):
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Audio engine
# ----------------------------

AUDIO_PRESETS: dict[str, str] = {
    "balanced": (
        "highpass=f=80,"
        "lowpass=f=11000,"
        "speechnorm=e=6:r=0.0001:l=1,"
        "afftdn=nf=-25,"
        "acompressor=threshold=-18dB:ratio=2:attack=5:release=50,"
        "dynaudnorm=f=150:g=7:p=0.95"
    ),
    "strong": (
        "highpass=f=100,"
        "lowpass=f=9000,"
        "speechnorm=e=8:r=0.0001:l=1,"
        "afftdn=nf=-30,"
        "acompressor=threshold=-20dB:ratio=3:attack=5:release=80,"
        "dynaudnorm=f=200:g=10:p=0.95"
    ),
}
DEFAULT_AUDIO_PRESET = "balanced"


def build_audio_filter(preset: str, gain_db: float = 0.0) -> str:
    p = (preset or DEFAULT_AUDIO_PRESET).strip().lower()
    chain = AUDIO_PRESETS.get(p, AUDIO_PRESETS[DEFAULT_AUDIO_PRESET])
    if abs(gain_db) > 0.01:
        chain = f"volume={gain_db}dB,{chain}"
    return chain


def _ffprobe_json(input_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(input_path),
    ]
    res = run(cmd, check=True)
    return json.loads(res.stdout or "{}")


def _get_duration_sec(meta: dict) -> Optional[float]:
    fmt = meta.get("format") or {}
    dur = fmt.get("duration")
    try:
        return float(dur) if dur is not None else None
    except Exception:
        return None


def pick_best_audio_map(input_path: Path) -> str:
    """
    Return mapping like '0:a:0' or '0:a:1'.
    Heuristic: highest bitrate; break ties with channels, then sample rate.
    """
    meta = _ffprobe_json(input_path)
    streams = meta.get("streams") or []

    audio_ord: list[tuple[int, dict]] = []
    ordinal = 0
    for s in streams:
        if s.get("codec_type") == "audio":
            audio_ord.append((ordinal, s))
            ordinal += 1

    if not audio_ord:
        return "0:a:0"

    def score(s: dict) -> tuple[int, int, int]:
        br = s.get("bit_rate")
        ch = s.get("channels")
        sr = s.get("sample_rate")
        try:
            br_i = int(br) if br is not None else 0
        except Exception:
            br_i = 0
        try:
            ch_i = int(ch) if ch is not None else 0
        except Exception:
            ch_i = 0
        try:
            sr_i = int(sr) if sr is not None else 0
        except Exception:
            sr_i = 0
        return (br_i, ch_i, sr_i)

    best_ordinal, _ = max(audio_ord, key=lambda t: score(t[1]))
    return f"0:a:{best_ordinal}"


def wav_preflight_ok(
    wav_path: Path, *, min_sec: float, expected_sec: Optional[float]
) -> tuple[bool, str]:
    if not wav_path.exists():
        return False, "wav missing"

    try:
        if wav_path.stat().st_size < 200_000:
            return False, "wav too small (likely wrong/empty track)"
    except Exception:
        pass

    meta = _ffprobe_json(wav_path)
    dur = _get_duration_sec(meta)
    if dur is None or dur <= 0.1:
        return False, "wav duration missing/zero"

    if dur < min_sec:
        return False, f"wav too short ({dur:.1f}s < {min_sec:.1f}s)"

    if expected_sec and expected_sec > 60 and dur < expected_sec * 0.60:
        return False, f"wav much shorter than input ({dur:.1f}s vs ~{expected_sec:.1f}s)"

    streams = meta.get("streams") or []
    a0 = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if a0:
        sr = str(a0.get("sample_rate") or "")
        ch = a0.get("channels")
        if sr and sr != "16000":
            return False, f"wav sample rate is {sr}, expected 16000"
        if ch is not None and int(ch) != 1:
            return False, f"wav channels is {ch}, expected 1"

    return True, "ok"


def extract_and_clean_audio(
    input_path: Path,
    out_wav: Path,
    *,
    no_clean: bool,
    audio_stream: Optional[str],
    audio_auto: bool,
    min_audio_sec: float,
    preset: str,
    gain_db: float,
) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    expected_sec: Optional[float]
    try:
        expected_sec = _get_duration_sec(_ffprobe_json(input_path))
    except Exception:
        expected_sec = None

    if audio_stream:
        map_sel = audio_stream
    elif audio_auto:
        map_sel = pick_best_audio_map(input_path)
    else:
        map_sel = "0:a:0"

    cmd_extract = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-thread_queue_size",
        "4096",
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        "-i",
        str(input_path),
        "-map",
        map_sel,
        "-vn",
        "-af",
        "aresample=async=1:first_pts=0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    run(cmd_extract, check=True)

    ok, reason = wav_preflight_ok(out_wav, min_sec=min_audio_sec, expected_sec=expected_sec)

    if not ok:
        bad = out_wav.with_suffix(".bad.wav")
        try:
            if bad.exists():
                bad.unlink()
            out_wav.replace(bad)
        except Exception:
            pass

        print(f"[subloom] audio preflight failed ({reason}) — retrying extraction (map {map_sel})")

        cmd_extract_fallback = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-thread_queue_size",
            "8192",
            "-analyzeduration",
            "200M",
            "-probesize",
            "200M",
            "-fflags",
            "+genpts+igndts",
            "-avoid_negative_ts",
            "make_zero",
            "-i",
            str(input_path),
            "-map",
            map_sel,
            "-vn",
            "-af",
            "aresample=async=1:first_pts=0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]
        run(cmd_extract_fallback, check=True)

        ok2, reason2 = wav_preflight_ok(out_wav, min_sec=min_audio_sec, expected_sec=expected_sec)
        if not ok2:
            raise RuntimeError(
                "Audio extraction still looks broken after fallback.\n"
                f"  input: {input_path}\n"
                f"  map:   {map_sel}\n"
                f"  reason: {reason2}\n"
                "Try: --audio-stream 0:a:1 (or another track)\n"
            )

    if no_clean:
        return out_wav

    clean_wav = out_wav.with_suffix(".clean.wav")
    af = build_audio_filter(preset=preset, gain_db=gain_db)

    cmd_clean = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(out_wav),
        "-af",
        af,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(clean_wav),
    ]
    run(cmd_clean, check=True)
    return clean_wav


# ----------------------------
# Small basic checks + helpers
# ----------------------------

def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s.strip()))


def ensure_tools() -> None:
    missing: list[str] = []
    for tool in ["ffmpeg", "ffprobe", "yt-dlp"]:
        if not have(tool):
            missing.append(tool)

    if not WCPP_BIN.exists():
        missing.append(f"whisper-cli ({WCPP_BIN})")
    if not WCPP_MODEL.exists():
        missing.append(f"ggml-large-v3 model ({WCPP_MODEL})")

    if missing:
        raise SystemExit("Missing requirements:\n" + "\n".join(f"- {m}" for m in missing))


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
        raise RuntimeError(f"yt-dlp said it downloaded to '{path}', but file not found")
    return path


def iter_media_files(folder: Path) -> Iterable[Path]:
    exts = {".mkv", ".mp4", ".webm", ".avi", ".mov", ".m4v", ".mp3", ".flac", ".ogg", ".wav"}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


# ----------------------------
# One-file pipeline
# ----------------------------

def _copy_text_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        dst.write_bytes(src.read_bytes())


def process_one(input_arg: str, workdir: Path, args) -> None:
    ensure_project_dirs()
    ensure_tools()
    t0 = time.perf_counter()

    if is_url(input_arg):
        media_path = yt_dlp_download(input_arg, workdir)
    else:
        media_path = Path(input_arg).expanduser()

    if not media_path.exists():
        raise SystemExit(f"Input not found: {media_path}")

    stem = media_path.stem

    # Routing:
    #   asr/           -> whisper.srt + final.srt
    #   out/<stem>/    -> wav + json + txt junk
    #   post/          -> compare log
    out_dir = DIR_OUT / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[subloom] {media_path}")

    wav = out_dir / f"{stem}.wav"
    clean = extract_and_clean_audio(
        media_path,
        wav,
        no_clean=args.no_clean,
        audio_stream=args.audio_stream,
        audio_auto=not args.no_audio_auto,
        min_audio_sec=args.min_audio_sec,
        preset=args.audio_preset,
        gain_db=args.audio_gain_db,
    )

    # Raw ASR output prefix lives in out/<stem>/
    outbase = out_dir / stem

    # Raw outputs in OUT
    whisper_srt_out = out_dir / f"{stem}.{WHISPER_TAG}.srt"
    whisper_json_out = out_dir / f"{stem}.{WHISPER_TAG}.json"
    kotoba_json = out_dir / f"{stem}.{KOTOBA_TAG}.json"

    # Keepers in ASR
    whisper_srt = DIR_ASR / f"{stem}.{WHISPER_TAG}.srt"
    final_srt = DIR_ASR / f"{stem}.{FINAL_TAG}.srt"

    # Debug log in POST (only if requested)
    compare_log: Optional[Path] = None
    if args.compare_log:
        compare_log = DIR_POST / f"{stem}.{COMPARE_TAG}.txt"

    # 1) Whisper
    if args.resume and whisper_srt.exists() and whisper_json_out.exists():
        print("[1/3] whisper.cpp large-v3 (resume)")
        # ensure we still have the ASR copy even if someone deleted it
        if not whisper_srt.exists() and whisper_srt_out.exists():
            _copy_text_file(whisper_srt_out, whisper_srt)
    else:
        print("[1/3] whisper.cpp large-v3 ...")
        run_whispercpp(
            clean,
            outbase,
            whisper_bin=WCPP_BIN,
            whisper_model=WCPP_MODEL,
            whisper_tag=WHISPER_TAG,
            max_srt_line_dur=MAX_SRT_LINE_DUR,
        )
        if not whisper_srt_out.exists():
            raise RuntimeError(f"Whisper SRT missing after run: {whisper_srt_out}")
        _copy_text_file(whisper_srt_out, whisper_srt)

    # 2) Kotoba
    if args.kotoba_ggml:
        model_path = Path(args.kotoba_ggml).expanduser()
    else:
        model_path = find_kotoba_ggml_model(KOTOBA_GGML_DIR)

    if not model_path:
        raise RuntimeError(f"No Kotoba GGML .bin found in {KOTOBA_GGML_DIR} (or pass --kotoba-ggml)")

    if args.resume and kotoba_json.exists():
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
        if not kotoba_json.exists():
            raise RuntimeError(f"Kotoba JSON missing after run: {kotoba_json}")

    # 3) Merge + optional Ollama
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
        whisper_srt=whisper_srt,
        kotoba_json=kotoba_json,
        out_final_srt=final_srt,
        out_compare_log=compare_log,
        ollama_cfg=ollama_cfg,
    )

    if not args.keep_wav:
        for p in (wav, wav.with_suffix(".clean.wav"), wav.with_suffix(".bad.wav")):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    dt = time.perf_counter() - t0
    mins = int(dt // 60)
    secs = dt % 60
    print(f"[subloom] Done: {final_srt}")
    print(f"[subloom] Finished in {mins}m {secs:.1f}s" if mins else f"[subloom] Finished in {secs:.1f}s")


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="subloom",
        description="Whisper timing SRT + Kotoba check + automatic final SRT + optional Ollama proofreading",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--workdir", default=str(DEFAULT_WORKDIR), help="where URL downloads go")
        p.add_argument("--kotoba-ggml", default=None, help=f"Path to Kotoba GGML .bin (default auto from {KOTOBA_GGML_DIR})")
        p.add_argument("--resume", action="store_true", help="Skip whisper/kotoba runs if outputs already exist")
        p.add_argument("--keep-wav", action="store_true", help="Keep extracted WAVs")
        p.add_argument("--no-clean", action="store_true", help="Skip ffmpeg filter pass")
        p.add_argument("--compare-log", action="store_true", help="Write compare/debug log")

        # Audio capture upgrades
        p.add_argument("--audio-stream", default=None, help="Force audio stream mapping (example: 0:a:1). Overrides auto-pick.")
        p.add_argument("--no-audio-auto", action="store_true", help="Disable auto audio stream selection.")
        p.add_argument("--min-audio-sec", type=float, default=20.0, help="If extracted WAV is shorter than this, treat it as broken and retry extraction.")
        p.add_argument("--audio-preset", choices=["balanced", "strong"], default=DEFAULT_AUDIO_PRESET, help="Audio preprocessing preset.")
        p.add_argument("--audio-gain-db", type=float, default=0.0, help="Optional gain applied before filtering (example: 4.0 or -2.0).")

        # Ollama
        p.add_argument("--ollama", action="store_true", help="Enable Ollama proofreading on FINAL subtitles")
        p.add_argument("--ollama-model", default="llama3.1:latest", help="Ollama model name")
        p.add_argument("--ollama-style", choices=["neutral", "anime", "formal"], default="neutral")
        p.add_argument("--ollama-window-sec", type=float, default=45.0)
        p.add_argument("--ollama-max-chars", type=int, default=1000)
        p.add_argument("--ollama-retries", type=int, default=1)
        p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
        p.add_argument("--ollama-no-skip-music", action="store_true")

    ap_run = sub.add_parser("run", help="Process one file path or URL")
    ap_run.add_argument("input", help="file path or URL")
    add_common(ap_run)

    ap_batch = sub.add_parser("batch", help="Process all media files in a folder (recursive)")
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
