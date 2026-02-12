from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from merge_subs import run
from settings import (
    AUDIO_PRESETS,
    DEFAULT_AUDIO_PRESET,
    AUDIO_DEFAULTS,
    RNNOISE_MODE_DEFAULT,
)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------


@dataclass
class AudioConfig:
    audio_stream: Optional[str] = None
    audio_auto: bool = True
    min_audio_sec: float = AUDIO_DEFAULTS.min_audio_sec
    no_clean: bool = False
    preset: str = AUDIO_DEFAULTS.preset
    gain_db: float = AUDIO_DEFAULTS.gain_db
    target_sr: int = AUDIO_DEFAULTS.target_sr
    target_ch: int = AUDIO_DEFAULTS.target_ch
    use_dynaudnorm_in_extract: bool = AUDIO_DEFAULTS.use_dynaudnorm_in_extract
    extract_dynaudnorm: str = AUDIO_DEFAULTS.extract_dynaudnorm
    extract_resample: str = AUDIO_DEFAULTS.extract_resample
    fallback_probesize: str = AUDIO_DEFAULTS.fallback_probesize
    fallback_analyzeduration: str = AUDIO_DEFAULTS.fallback_analyzeduration

    # RNNoise-style denoise via ffmpeg's arnndn filter
    use_rnnoise: bool = True

    # RNNoise model path is auto-resolved (and required) unless you explicitly pass one.
    # You can override with: export SUBLOOM_RNNOISE_MODEL=/path/to/std.rnnn
    rnnoise_model: Optional[str] = None

    # fixed: single mix for whole file
    # chunk: per-chunk adaptive mix (better for anime with BGM vs quiet dialogue tails)
    rnnoise_mode: str = RNNOISE_MODE_DEFAULT  # fixed|chunk

    # Used when rnnoise_mode == "fixed"
    rnnoise_mix: float = 0.51

    # Used when rnnoise_mode == "chunk"
    rnnoise_chunk_s: float = 20.0
    rnnoise_analysis_frame_ms: int = 30
    rnnoise_mix_min: float = 0.28
    rnnoise_mix_max: float = 0.80


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


def ffmpeg_has_filter(name: str) -> bool:
    try:
        rr = run(["ffmpeg", "-hide_banner", "-filters"], check=True)
        txt = (rr.stdout or "") + "\n" + (rr.stderr or "")
        return name in txt
    except Exception:
        return False


def build_audio_filter(
    preset: str,
    gain_db: float = 0.0,
    *,
    use_rnnoise: bool = False,
    rnnoise_model: Optional[str] = None,
    rnnoise_mix: float = 0.51,
) -> str:
    p = (preset or DEFAULT_AUDIO_PRESET).strip().lower()
    chain = AUDIO_PRESETS.get(p, AUDIO_PRESETS[DEFAULT_AUDIO_PRESET])

    # Keep existing behavior: gain first
    if abs(gain_db) > 0.01:
        chain = f"volume={gain_db}dB,{chain}"

    # RNNoise denoise first, then rest of chain
    if use_rnnoise:
        if rnnoise_model:
            chain = f"arnndn=m='{rnnoise_model}':mix={rnnoise_mix},{chain}"
        else:
            chain = f"arnndn=mix={rnnoise_mix},{chain}"

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


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _norm_lang(tag: str) -> str:
    t = (tag or "").strip().lower()
    if t in ("jp", "jpn", "ja-jp"):
        return "ja"
    return t


# ------------------------------------------------------------
# Audio track selection
# ------------------------------------------------------------


def pick_best_audio_map(input_path: Path) -> str:
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

    def get_tags(s: dict) -> dict:
        t = s.get("tags")
        return t if isinstance(t, dict) else {}

    def lang_of(s: dict) -> str:
        tags = get_tags(s)
        return _norm_lang(str(tags.get("language") or ""))

    def title_of(s: dict) -> str:
        tags = get_tags(s)
        for k in ("title", "handler_name", "NAME", "name"):
            v = tags.get(k)
            if v:
                return str(v)
        return ""

    def is_commentary(s: dict) -> bool:
        t = (title_of(s) or "").lower()
        return any(x in t for x in ("commentary", "director", "descriptive"))

    def ch_score(ch: int) -> int:
        if ch == 2:
            return 30
        if ch == 1:
            return 25
        if ch == 6:
            return 5
        return max(0, 20 - abs(ch - 2) * 3)

    def score(s: dict) -> tuple[int, int, int, int]:
        br_i = _safe_int(s.get("bit_rate"), 0)
        sr_i = _safe_int(s.get("sample_rate"), 0)
        ch_i = _safe_int(s.get("channels"), 0)

        lang = lang_of(s)
        lang_boost = 50 if lang == "ja" else 0
        comm_penalty = -25 if is_commentary(s) else 0
        ch_boost = ch_score(ch_i)

        return (lang_boost, comm_penalty, ch_boost, br_i + sr_i)

    best_ordinal, _ = max(audio_ord, key=lambda t: score(t[1]))
    return f"0:a:{best_ordinal}"


# ------------------------------------------------------------
# Preflight check
# ------------------------------------------------------------


def wav_preflight_ok(
    wav_path: Path,
    *,
    min_sec: float,
    expected_sec: Optional[float],
    expected_sr: int,
    expected_ch: int,
) -> tuple[bool, str]:
    if not wav_path.exists():
        return False, "wav missing"

    meta = _ffprobe_json(wav_path)
    dur = _get_duration_sec(meta)
    if dur is None or dur < min_sec:
        return False, "wav too short"

    # Silence check
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-t",
            "25",
            "-i",
            str(wav_path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ]
        rr = run(cmd, check=True)
        st = rr.stderr or ""
        m_mean = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", st)
        m_max = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", st)

        if m_mean and m_max:
            mean_db = float(m_mean.group(1))
            max_db = float(m_max.group(1))
            if mean_db <= -55.0 and max_db <= -40.0:
                return False, "wav near-silent"
    except Exception:
        pass

    return True, "ok"


# ------------------------------------------------------------
# Extraction + cleaning
# ------------------------------------------------------------


def extract_audio_stable(input_path: Path, out_wav: Path, cfg: AudioConfig) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if cfg.audio_stream:
        map_sel = cfg.audio_stream
    elif cfg.audio_auto:
        map_sel = pick_best_audio_map(input_path)
    else:
        map_sel = "0:a:0"

    af = cfg.extract_resample
    if cfg.use_dynaudnorm_in_extract:
        af = f"{af},{cfg.extract_dynaudnorm}"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        map_sel,
        "-vn",
        "-af",
        af,
        "-ac",
        str(cfg.target_ch),
        "-ar",
        str(cfg.target_sr),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    run(cmd, check=True)

    ok, reason = wav_preflight_ok(
        out_wav,
        min_sec=cfg.min_audio_sec,
        expected_sec=None,
        expected_sr=cfg.target_sr,
        expected_ch=cfg.target_ch,
    )
    if not ok:
        raise RuntimeError(f"Audio extraction failed: {reason}")

    return out_wav


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _choose_chunk_mix_from_pcm16(
    wav_path: Path, *, frame_ms: int, mix_min: float, mix_max: float
) -> float:
    """Very fast heuristic to choose RNNoise mix for a chunk.

    Uses short-frame RMS variability:
    - speech tends to have higher RMS variance (bursty)
    - music/noise tends to be more constant

    We push mix higher when energy is high *and* variability is low.
    """
    try:
        import wave
        import array

        with wave.open(str(wav_path), "rb") as w:
            n_channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            fr = w.getframerate()
            n_frames = w.getnframes()

            if sampwidth != 2 or n_frames == 0:
                return (mix_min + mix_max) / 2.0

            raw = w.readframes(n_frames)
        a = array.array("h")
        a.frombytes(raw)

        # mix down to mono in-place-ish
        if n_channels == 2:
            # take average of L/R (cheap)
            mono = [(a[i] + a[i + 1]) // 2 for i in range(0, len(a) - 1, 2)]
        else:
            mono = a.tolist()

        frame_n = max(1, int(fr * (frame_ms / 1000.0)))
        rms_vals = []
        # sample at most ~8 seconds worth of frames for speed
        max_samples = min(len(mono), fr * 8)
        step = frame_n
        for s in range(0, max_samples, step):
            chunk = mono[s : s + frame_n]
            if not chunk:
                break
            # RMS
            acc = 0.0
            for v in chunk:
                acc += float(v) * float(v)
            rms = (acc / max(1, len(chunk))) ** 0.5
            rms_vals.append(rms)

        if not rms_vals:
            return (mix_min + mix_max) / 2.0

        mean = sum(rms_vals) / len(rms_vals)
        var = sum((x - mean) ** 2 for x in rms_vals) / len(rms_vals)
        std = var**0.5

        # normalize mean roughly into [0,1] using int16 range
        mean_norm = _clamp(mean / 12000.0, 0.0, 1.0)
        # coefficient of variation, clipped
        cv = _clamp(std / (mean + 1e-6), 0.0, 2.0)
        cv_norm = _clamp(cv / 1.2, 0.0, 1.0)

        # Higher when energy high and variability low
        score = _clamp(mean_norm * (1.0 - cv_norm), 0.0, 1.0)

        return mix_min + score * (mix_max - mix_min)
    except Exception:
        return (mix_min + mix_max) / 2.0


def _segment_wav(input_wav: Path, seg_dir: Path, chunk_s: float) -> list[Path]:
    seg_dir.mkdir(parents=True, exist_ok=True)
    pattern = seg_dir / "chunk_%05d.wav"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_wav),
        "-f",
        "segment",
        "-segment_time",
        f"{chunk_s}",
        "-reset_timestamps",
        "1",
        "-c:a",
        "pcm_s16le",
        str(pattern),
    ]
    run(cmd, check=True)
    return sorted(seg_dir.glob("chunk_*.wav"))


def _concat_wavs(wavs: list[Path], out_wav: Path) -> None:
    if not wavs:
        raise RuntimeError("No chunks to concatenate.")
    lst = out_wav.with_suffix(".concat.txt")
    lst.write_text(
        "\n".join([f"file '{p.as_posix()}'" for p in wavs]) + "\n", encoding="utf-8"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(lst),
        "-c",
        "copy",
        str(out_wav),
    ]
    run(cmd, check=True)
    try:
        lst.unlink()
    except Exception:
        pass


def extract_and_clean_audio(
    input_path: Path,
    out_wav: Path,
    cfg: AudioConfig,
    *,
    out_wav_stereo: Optional[Path] = None,
) -> Path:
    raw_wav = extract_audio_stable(input_path, out_wav, cfg)

    if cfg.no_clean:
        return raw_wav

    clean_wav = out_wav.with_name(out_wav.stem + ".clean.wav")

    # RNNoise is a requirement (quality backbone) if enabled.
    use_rn = bool(cfg.use_rnnoise) and ffmpeg_has_filter("arnndn")
    if cfg.use_rnnoise and not use_rn:
        raise RuntimeError(
            "ffmpeg filter 'arnndn' not available, but RNNoise is enabled/required."
        )

    rn_model = cfg.rnnoise_model
    if use_rn and not rn_model:
        # auto-resolve from settings (required)
        from settings import resolve_rnnoise_model

        rn_model = str(resolve_rnnoise_model())

    # Chunk-adaptive RNNoise (per-chunk mix), then apply the rest of the chain.
    if use_rn and (cfg.rnnoise_mode or "").strip().lower() == "chunk":
        seg_dir = out_wav.with_suffix("").with_name(out_wav.stem + ".rnnoise_chunks")
        den_dir = out_wav.with_suffix("").with_name(out_wav.stem + ".rnnoise_denoised")
        try:
            chunks = _segment_wav(raw_wav, seg_dir, cfg.rnnoise_chunk_s)
            if not chunks:
                raise RuntimeError("RNNoise chunking produced no chunks.")

            denoised: list[Path] = []
            for ch in chunks:
                mix = _choose_chunk_mix_from_pcm16(
                    ch,
                    frame_ms=cfg.rnnoise_analysis_frame_ms,
                    mix_min=cfg.rnnoise_mix_min,
                    mix_max=cfg.rnnoise_mix_max,
                )
                den_dir.mkdir(parents=True, exist_ok=True)
                out_ch = den_dir / ch.name
                af_rn = f"arnndn=m='{rn_model}':mix={mix}"
                cmd_rn = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(ch),
                    "-af",
                    af_rn,
                    "-ac",
                    str(cfg.target_ch),
                    "-ar",
                    str(cfg.target_sr),
                    str(out_ch),
                ]
                run(cmd_rn, check=True)
                denoised.append(out_ch)

            pre_wav = out_wav.with_name(out_wav.stem + ".rnnoise.wav")
            _concat_wavs(denoised, pre_wav)

            # Now run your normal preset chain (WITHOUT arnndn) on the rnnoise output.
            af = build_audio_filter(
                cfg.preset,
                cfg.gain_db,
                use_rnnoise=False,
                rnnoise_model=None,
                rnnoise_mix=cfg.rnnoise_mix,
            )
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(pre_wav),
                "-af",
                af,
                "-ac",
                str(cfg.target_ch),
                "-ar",
                str(cfg.target_sr),
                str(clean_wav),
            ]
            run(cmd, check=True)
        finally:
            # keep chunk dirs only if you want to debug
            pass
    else:
        af = build_audio_filter(
            cfg.preset,
            cfg.gain_db,
            use_rnnoise=use_rn,
            rnnoise_model=rn_model,
            rnnoise_mix=cfg.rnnoise_mix,
        )

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw_wav),
            "-af",
            af,
            "-ac",
            str(cfg.target_ch),
            "-ar",
            str(cfg.target_sr),
            str(clean_wav),
        ]
        run(cmd, check=True)

    if out_wav_stereo:
        cmd_stereo = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw_wav),
            "-ac",
            "2",
            "-ar",
            str(cfg.target_sr),
            str(out_wav_stereo),
        ]
        run(cmd_stereo, check=True)

    return clean_wav
