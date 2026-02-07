from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from merge_subs import run, seconds_to_srt_time

# NOTE: Import settings lazily inside export_srt_from_whisper_json()
# to avoid any annoying import-order surprises.


@dataclass
class WhisperPaths:
    json_path: Path
    srt_path: Path
    txt_path: Optional[Path] = None


_RE_WS = re.compile(r"\s+")


def _clean_text_for_split(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # keep JP readable; just normalize whitespace
    s = _RE_WS.sub(" ", s)
    return s.strip()


def _split_by_preferred_punct(text: str) -> list[str]:
    """
    First pass: split at sentence-ish punctuation when possible.
    Keeps punctuation attached.
    """
    t = (text or "").strip()
    if not t:
        return []

    out: list[str] = []
    buf: list[str] = []
    enders = set("。！？!?")

    for ch in t:
        buf.append(ch)
        if ch in enders:
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []

    tail = "".join(buf).strip()
    if tail:
        out.append(tail)

    return out


def _hard_split(text: str, max_chars: int) -> list[str]:
    """
    Last-resort split: chunk by character count.
    """
    t = (text or "").strip()
    if not t:
        return []
    if max_chars <= 1:
        return [t]
    return [
        t[i : i + max_chars].strip()
        for i in range(0, len(t), max_chars)
        if t[i : i + max_chars].strip()
    ]


def _split_to_max_chars(text: str, max_chars: int) -> list[str]:
    """
    Split text into multiple caption-sized chunks with a preference order:
      1) sentence enders: 。！？!? (kept)
      2) soft breaks: 、, (kept)
      3) hard split by length
    """
    t = _clean_text_for_split(text)
    if not t:
        return []

    if len(t) <= max_chars:
        return [t]

    # 1) Sentence-ish split
    parts = _split_by_preferred_punct(t)
    if not parts:
        parts = [t]

    # Now pack parts into <= max_chars chunks
    packed: list[str] = []
    cur = ""

    def flush():
        nonlocal cur
        if cur.strip():
            packed.append(cur.strip())
        cur = ""

    for p in parts:
        p = p.strip()
        if not p:
            continue

        if not cur:
            cur = p
            continue

        if len(cur) + len(p) <= max_chars:
            cur = cur + p
            continue

        flush()
        cur = p

    flush()

    # 2) If any packed chunk is still too long, try splitting on commas
    fixed: list[str] = []
    soft_breaks = ("、", ",", "，")

    for chunk in packed:
        if len(chunk) <= max_chars:
            fixed.append(chunk)
            continue

        # split by soft breaks, keep delimiter
        buf = ""
        for ch in chunk:
            buf += ch
            if ch in soft_breaks and len(buf) >= max(6, max_chars // 3):
                fixed.append(buf.strip())
                buf = ""
        if buf.strip():
            fixed.append(buf.strip())

    # 3) Final guarantee (hard split)
    final: list[str] = []
    for chunk in fixed:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            final.extend(_hard_split(chunk, max_chars))

    return [x for x in (s.strip() for s in final) if x]


def _wrap_lines(text: str, max_line_chars: int) -> str:
    """
    Wrap into 1–2 lines if needed.
    Uses punctuation as a nicer break when possible.
    """
    t = (text or "").strip()
    if not t:
        return ""

    if max_line_chars <= 0 or len(t) <= max_line_chars:
        return t

    # Try to find a good split point near the middle
    target = min(len(t) - 1, max_line_chars)
    candidates = []

    # Prefer these as line breaks
    preferred = set("、,，。！？!?")

    for i in range(max(1, target - 8), min(len(t) - 1, target + 8) + 1):
        if t[i] in preferred:
            candidates.append(i + 1)  # split AFTER punctuation

    if candidates:
        cut = min(candidates, key=lambda x: abs(x - target))
        a = t[:cut].strip()
        b = t[cut:].strip()
        if a and b:
            return a + "\n" + b

    # Fallback: straight cut
    a = t[:target].strip()
    b = t[target:].strip()
    if a and b:
        return a + "\n" + b
    return t


def export_srt_from_whisper_json(
    json_path: Path, srt_path: Path, max_line_dur: float
) -> None:
    """
    whisper.cpp outputs a JSON 'transcription' list with millisecond offsets.
    This writes a normal SRT while:
      - shifting timestamps later (fixes "subs come too early")
      - splitting long captions (fixes "wall of text")
      - wrapping to 1–2 lines
    """
    from settings import (
        SRT_SHIFT_START_S,
        SRT_SHIFT_END_S,
        SRT_MIN_CAPTION_DUR_S,
        SRT_MAX_CHARS_PER_CAPTION,
        SRT_MAX_CHARS_PER_LINE,
    )

    raw = json_path.read_bytes()
    # Whisper JSON should be UTF-8, but some builds output odd bytes.
    # Decode safely so no crashes.
    text = raw.decode("utf-8", errors="replace")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # fallback: ignore bad bytes completely
        text2 = raw.decode("utf-8", errors="ignore")
        data = json.loads(text2)

    transcription = data.get("transcription")
    if not isinstance(transcription, list):
        keys = (
            ", ".join(list(data.keys())[:30])
            if isinstance(data, dict)
            else "<not a dict>"
        )
        raise RuntimeError(
            f"Expected 'transcription' list in whisper JSON. Top-level keys: {keys}"
        )

    out_lines: list[str] = []
    idx = 1

    for item in transcription:
        if not isinstance(item, dict):
            continue

        raw_text = (item.get("text") or "").strip()
        if not raw_text:
            continue

        offsets = item.get("offsets") or {}
        start_ms = offsets.get("from")
        end_ms = offsets.get("to")
        if start_ms is None or end_ms is None:
            continue

        start_s = float(start_ms) / 1000.0
        end_s = float(end_ms) / 1000.0
        if end_s <= start_s:
            end_s = start_s + 1.2

        # Shift later if subs feel early
        # (start shift slightly bigger than end shift to stop early pop-in)
        if SRT_SHIFT_START_S or SRT_SHIFT_END_S:
            start_s = max(0.0, start_s + float(SRT_SHIFT_START_S))
            end_s = max(start_s + 0.10, end_s + float(SRT_SHIFT_END_S))

        # Cap extreme long durations (safety)
        dur = end_s - start_s
        if dur > max_line_dur:
            end_s = start_s + float(max_line_dur)
            dur = end_s - start_s

        # Split long text into multiple caption entries
        chunks = _split_to_max_chars(raw_text, int(SRT_MAX_CHARS_PER_CAPTION) or 999999)
        if not chunks:
            continue

        # Time-split: allocate duration proportionally to text length
        # (simple + stable, and it avoids "one chunk gets 0.1s")
        lens = [max(1, len(c)) for c in chunks]
        total_len = sum(lens)

        # If whisper gave us an ultra-short duration but we have multiple chunks,
        # enforce a minimum-per-caption where possible.
        min_d = float(SRT_MIN_CAPTION_DUR_S)
        if len(chunks) == 1:
            starts_ends = [(start_s, end_s)]
        else:
            desired = len(chunks) * min_d
            if dur < desired:
                # We can't magically create time, so we distribute what we have,
                # but we still keep each chunk non-zero.
                min_d = max(0.28, dur / len(chunks))

            cur_t = start_s
            starts_ends = []
            for i, (chunk_text, chunk_len) in enumerate(zip(chunks, lens)):
                # last chunk gets whatever remains (prevents drift)
                if i == len(chunks) - 1:
                    s0 = cur_t
                    e0 = end_s
                else:
                    share = dur * (chunk_len / total_len)
                    share = max(min_d, share)
                    s0 = cur_t
                    e0 = min(end_s, s0 + share)

                if e0 <= s0:
                    e0 = min(end_s, s0 + 0.30)

                starts_ends.append((s0, e0))
                cur_t = e0

        for chunk_text, (s0, e0) in zip(chunks, starts_ends):
            txt = _wrap_lines(chunk_text, int(SRT_MAX_CHARS_PER_LINE) or 999999)
            if not txt.strip():
                continue

            out_lines.append(str(idx))
            out_lines.append(f"{seconds_to_srt_time(s0)} --> {seconds_to_srt_time(e0)}")
            out_lines.append(txt)
            out_lines.append("")
            idx += 1

    srt_path.write_text("\n".join(out_lines), encoding="utf-8")


def run_whispercpp(
    clean_wav: Path,
    outbase: Path,
    *,
    whisper_bin: Path,
    whisper_model: Path,
    whisper_tag: str,
    max_srt_line_dur: float,
    threads: int = 8,
    beam_size: int = 2,
    best_of: int = 2,
    device: str | None = "0",
) -> WhisperPaths:
    """
    Pass 1: Whisper (timing source).
    Runs whisper.cpp, writes JSON + TXT, then generates the timing SRT from JSON.
    Outputs are written next to outbase using names like:
      <outbase>.<whisper_tag>.json/.txt/.srt
    """
    out_prefix = str(outbase.parent / f"{outbase.name}.{whisper_tag}")

    cmd: list[str] = [
        str(whisper_bin),
        "-m",
        str(whisper_model),
        "-f",
        str(clean_wav),
        "-l",
        "ja",
        "-t",
        str(max(1, int(threads))),
        "-bs",
        str(max(1, int(beam_size))),
        "-bo",
        str(max(1, int(best_of))),
        "-mc",
        "3",
    ]

    # GPU (optional)
    if device is not None:
        cmd += ["-dev", str(device), "-fa"]

    cmd += ["-otxt", "-oj", "-of", out_prefix]

    print("[subloom] whisper cmd:", " ".join(cmd))
    run(cmd, check=True)

    json_path = Path(out_prefix + ".json")
    txt_path = Path(out_prefix + ".txt")
    srt_path = Path(out_prefix + ".srt")

    if not json_path.exists():
        raise RuntimeError(f"Whisper JSON not found after run: {json_path}")

    # generate SRT from JSON
    export_srt_from_whisper_json(json_path, srt_path, max_srt_line_dur)

    if not srt_path.exists():
        raise RuntimeError(f"Whisper SRT not found after export: {srt_path}")

    return WhisperPaths(
        json_path=json_path,
        srt_path=srt_path,
        txt_path=txt_path if txt_path.exists() else None,
    )
