from __future__ import annotations
from merge_subs import run

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class KotobaPaths:
    json_path: Path
    txt_path: Path


def find_kotoba_ggml_model(models_dir: Path) -> Optional[Path]:
    """
    Picks a Kotoba GGML .bin from a folder.
    Preference order:
    - q5_0 if present (good speed/quality sweet spot)
    - otherwise biggest file
    """
    if not models_dir.exists():
        return None
    candidates = [p for p in models_dir.glob("*.bin") if p.is_file()]
    if not candidates:
        return None

    for p in candidates:
        if "q5_0" in p.name:
            return p

    candidates.sort(key=lambda x: x.stat().st_size, reverse=True)
    return candidates[0]


def run_kotoba_whispercpp(
    clean_wav: Path,
    outbase: Path,
    kotoba_model_path: Path,
    *,
    whisper_bin: Path,
    kotoba_tag: str,
) -> KotobaPaths:
    """
    Pass 2: Kotoba (accuracy checker).
    Runs Kotoba via whisper.cpp binary, writes JSON + TXT.
    We intentionally do NOT generate a Kotoba SRT — Whisper owns timing.
    """
    if not kotoba_model_path.exists():
        raise RuntimeError(f"Kotoba GGML model not found: {kotoba_model_path}")

    out_prefix = str(outbase) + f".{kotoba_tag}"

    cmd = [
        str(whisper_bin),
        "-m",
        str(kotoba_model_path),
        "-f",
        str(clean_wav),
        "-l",
        "ja",
        "-dev",
        "0",
        "-mc",
        "3",
        "-otxt",
        "-oj",
        "-of",
        out_prefix,
    ]

    print("[subloom] kotoba(ggml) cmd:", " ".join(cmd))
    run(cmd, check=True)

    kotoba_txt = Path(out_prefix + ".txt")
    kotoba_json = Path(out_prefix + ".json")

    if not kotoba_txt.exists():
        raise RuntimeError(f"Kotoba TXT not found after run: {kotoba_txt}")
    if not kotoba_json.exists():
        raise RuntimeError(f"Kotoba JSON not found after run: {kotoba_json}")

    return KotobaPaths(json_path=kotoba_json, txt_path=kotoba_txt)
