from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final
import os

# ----------------------------
# Project paths
# ----------------------------

APP_DIR: Final[Path] = Path(__file__).resolve().parent

DIR_ASR: Final[Path] = APP_DIR / "asr"
DIR_OUT: Final[Path] = APP_DIR / "out"
DIR_POST: Final[Path] = APP_DIR / "post"
DIR_WORK: Final[Path] = APP_DIR / "work"
DIR_MODELS: Final[Path] = APP_DIR / "models"

# whisper.cpp stored inside subloom/models/whisper.cpp
WCPP_BASE: Final[Path] = DIR_MODELS / "whisper.cpp"
WCPP_BIN: Final[Path] = WCPP_BASE / "build/bin/whisper-cli"

# whisper.cpp models live here
WCPP_MODELS_DIR: Final[Path] = WCPP_BASE / "models"

# Default Whisper model name. This is only a *preference*.
DEFAULT_WCPP_MODEL_NAME: Final[str] = "ggml-large-v3.bin"

# Backwards-compatible constant (so older code doesn't break)
WCPP_MODEL: Final[Path] = WCPP_MODELS_DIR / DEFAULT_WCPP_MODEL_NAME

# Kotoba models stored inside subloom/models/kotoba/
KOTOBA_GGML_DIR: Final[Path] = DIR_MODELS / "kotoba"

DEFAULT_WORKDIR: Final[Path] = DIR_WORK

# Tags / naming
WHISPER_TAG: Final[str] = "whisper"
KOTOBA_TAG: Final[str] = "kotoba"
FINAL_TAG: Final[str] = "final"
COMPARE_TAG: Final[str] = "compare"

# ----------------------------
# Subtitle / VAD defaults
# ----------------------------
SRT_NO_SPLIT_UNDER_S: Final[float] = 1.6
VAD_FRAME_MS: Final[int] = 30
VAD_DBFS_THRESH: Final[float] = -34.0
VAD_MIN_REGION_S: Final[float] = 0.25
VAD_PAD_S: Final[float] = 0.22
VAD_MERGE_GAP_S: Final[float] = 0.25
VAD_MAX_WINDOW_S: Final[float] = 12.0
VAD_MAX_TOTAL_DECODE_S: Final[float] = 240.0
CONF_TIMING_LOWCONF: Final[float] = 0.18
CONF_TIMING_MAX_EXTRA_END_S: Final[float] = 0.18

# ----------------------------
# RNNoise defaults
# ----------------------------
RNNOISE_REQUIRED: Final[bool] = True
RNNOISE_MODE_DEFAULT: Final[str] = "chunk"  # fixed|chunk
RNNOISE_MIX_DEFAULT: Final[float] = 0.51
RNNOISE_CHUNK_S_DEFAULT: Final[float] = 20.0
RNNOISE_MIX_MIN_DEFAULT: Final[float] = 0.28
RNNOISE_MIX_MAX_DEFAULT: Final[float] = 0.80
RNNOISE_ANALYSIS_FRAME_MS_DEFAULT: Final[int] = 30


def resolve_rnnoise_model() -> Path:
    """Locate RNNoise .rnnn model. Required if RNNOISE_REQUIRED is True."""
    env = os.environ.get("SUBLOOM_RNNOISE_MODEL")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
        raise RuntimeError(f"RNNoise model not found at: {p}")

    models_dir = APP_DIR / "models" / "arnndn-models"
    if not models_dir.exists():
        raise RuntimeError(
            f"RNNoise models directory not found: {models_dir}\n"
            "Place a .rnnn model in subloom/models/arnndn-models/ (e.g. std.rnnn)."
        )

    cands = sorted(models_dir.glob("*.rnnn"))
    if not cands:
        raise RuntimeError(
            f"No .rnnn RNNoise models found in: {models_dir}\n"
            "Download/copy a model there (e.g. std.rnnn)."
        )

    return cands[0]


# ----------------------------
# SRT formatting limits
# ----------------------------

# Safety cap: a single subtitle entry will never exceed this duration.
MAX_SRT_LINE_DUR: Final[float] = 10.0

# If subs feel early, increase this a bit (0.12–0.25 is usually the sweet spot).
# This shifts BOTH start + end later to preserve the original duration.
SRT_SHIFT_S: Final[float] = 0.18

# Minimum caption duration - balanced to prevent cutoffs without being too long
SRT_MIN_CAPTION_DUR_S: Final[float] = 0.70

# Generous character limits to reduce aggressive splitting
SRT_MAX_CHARS_PER_CAPTION: Final[int] = 50

# Also wrap text inside a caption so it doesn't become one mega-line.
SRT_MAX_CHARS_PER_LINE: Final[int] = 25

# Only slight adjustments needed since speech estimation is (probably??) now accurate
SRT_SHIFT_START_S = 0.0  # Minimal start adjustment
SRT_SHIFT_END_S = 0.0  # Minimal end adjustment

# Small buffer ensures subtitles don't feel cut off
SRT_END_PADDING_S: Final[float] = 0.10

# Prevents previous subtitle from bleeding into next one
SRT_MIN_GAP_S: Final[float] = 0.0


def ensure_project_dirs() -> None:
    for d in (DIR_ASR, DIR_OUT, DIR_POST, DIR_WORK, DIR_MODELS):
        d.mkdir(parents=True, exist_ok=True)


def resolve_whisper_model() -> Path:
    """
    Auto-detect whisper.cpp model (no CLI required).

    Priority:
      1) $SUBLOOM_WCPP_MODEL (full path or filename inside whisper.cpp/models)
         - also accepts 'medium', 'ggml-medium', etc.
      2) Prefer common models if present (large-v3 -> large -> medium -> small -> base -> tiny)
      3) Fall back to DEFAULT_WCPP_MODEL_NAME if it exists
      4) Fall back to first ggml-*.bin found
    """
    import os

    models_dir = WCPP_MODELS_DIR

    raw = (os.environ.get("SUBLOOM_WCPP_MODEL") or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if p.is_file():
            return p

        p2 = models_dir / raw
        if p2.is_file():
            return p2

        short = raw
        if not short.startswith("ggml-"):
            short = f"ggml-{short}"
        if not short.endswith(".bin"):
            short = f"{short}.bin"
        p3 = models_dir / short
        if p3.is_file():
            return p3

        raise FileNotFoundError(f"SUBLOOM_WCPP_MODEL set but model not found: {raw}")

    preferred = [
        "ggml-large-v3.bin",
        "ggml-large.bin",
        "ggml-medium.bin",
        "ggml-small.bin",
        "ggml-base.bin",
        "ggml-tiny.bin",
    ]
    for name in preferred:
        p = models_dir / name
        if p.is_file():
            return p

    p = models_dir / DEFAULT_WCPP_MODEL_NAME
    if p.is_file():
        return p

    candidates = sorted(models_dir.glob("ggml-*.bin"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No whisper models found in: {models_dir}")


# ----------------------------
# Audio defaults
# ----------------------------

AUDIO_PRESETS: Final[dict[str, str]] = {
    "anime": (
        "highpass=f=80,"
        "lowpass=f=11500,"
        "afftdn=nf=-24,"
        "acompressor=threshold=-20dB:ratio=4:attack=2:release=120:makeup=6,"
        "dynaudnorm=f=250:g=14:p=0.85:m=10,"
        "alimiter=limit=0.97"
    ),
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

DEFAULT_AUDIO_PRESET: Final[str] = "balanced"


@dataclass
class AudioDefaults:
    target_sr: int = 16000
    target_ch: int = 1

    # Extract stability
    use_dynaudnorm_in_extract: bool = True
    extract_resample: str = "aresample=async=1:first_pts=0"
    extract_dynaudnorm: str = "dynaudnorm=f=150:g=15"

    # Fallback probe settings
    fallback_probesize: str = "200M"
    fallback_analyzeduration: str = "200M"

    # Preflight
    min_audio_sec: float = 20.0

    # Cleaning
    preset: str = DEFAULT_AUDIO_PRESET
    gain_db: float = 0.0


AUDIO_DEFAULTS: Final[AudioDefaults] = AudioDefaults()
