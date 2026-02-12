from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from merge_subs import run, seconds_to_srt_time


@dataclass
class WhisperPaths:
    json_path: Path
    srt_path: Path
    txt_path: Optional[Path] = None


_RE_WS = re.compile(r"\s+")

# Common filler phrases from BGM/outros
_FILLER_PHRASES = {
    "ご視聴ありがとうございました",
}


def _is_filler_phrase(text: str) -> bool:
    """Check if text is a common filler/outro phrase."""
    return text.strip() in _FILLER_PHRASES


def _clean_text(s: str) -> str:
    """Normalize whitespace."""
    s = (s or "").strip()
    if not s:
        return ""
    return _RE_WS.sub(" ", s).strip()


def _estimate_reading_time(text: str) -> float:
    """
    Estimate minimum time needed to read Japanese text.
    Longer text gets proportionally more time (people slow down for long sentences).
    """
    clean = _clean_text(text)
    if not clean:
        return 0.4

    char_count = len(clean)

    # Base rate: 6 chars/sec for short text
    # But slow down for longer text (harder to process)
    if char_count <= 15:
        rate = 6.0  # Fast, easy to read
    elif char_count <= 30:
        rate = 5.5  # Medium
    else:
        rate = 5.0  # Slower for long sentences

    return max(0.4, char_count / rate)


def _split_text_smart(text: str, max_chars: int) -> list[str]:
    """
    Split text at natural boundaries.
    Priority: 。！？!? -> 、， -> character limit
    """
    text = _clean_text(text)
    if not text or len(text) <= max_chars:
        return [text] if text else []

    # Split at sentence enders
    sentence_enders = set("。！？!?")
    parts = []
    current = ""

    for char in text:
        current += char
        if char in sentence_enders:
            if len(current) <= max_chars:
                parts.append(current.strip())
                current = ""
            else:
                parts.append(current.strip())
                current = ""

    if current.strip():
        parts.append(current.strip())

    if not parts:
        parts = [text]

    # Split any parts that are still too long
    final_parts = []
    for part in parts:
        if len(part) <= max_chars:
            final_parts.append(part)
        else:
            # Split at commas
            soft_breaks = "、，,"
            sub_parts = []
            buf = ""
            for char in part:
                buf += char
                if char in soft_breaks and len(buf) >= 10:
                    sub_parts.append(buf.strip())
                    buf = ""
            if buf.strip():
                sub_parts.append(buf.strip())

            # Hard split if still too long
            for sub in sub_parts:
                if len(sub) <= max_chars:
                    final_parts.append(sub)
                else:
                    for i in range(0, len(sub), max_chars):
                        chunk = sub[i : i + max_chars].strip()
                        if chunk:
                            final_parts.append(chunk)

    return [p for p in final_parts if p]


def _wrap_lines(text: str, max_line_len: int) -> str:
    """Wrap text into 1-2 lines."""
    text = _clean_text(text)
    if not text or len(text) <= max_line_len:
        return text

    # Try splitting at punctuation near middle
    mid = len(text) // 2
    search_range = 8

    punct = set("。！？!?、，,")
    best_split = None
    best_distance = float("inf")

    for i in range(max(0, mid - search_range), min(len(text), mid + search_range)):
        if i < len(text) and text[i] in punct:
            distance = abs(i - mid)
            if distance < best_distance:
                best_distance = distance
                best_split = i + 1

    if best_split:
        line1 = text[:best_split].strip()
        line2 = text[best_split:].strip()
        if line1 and line2:
            return f"{line1}\n{line2}"

    # Fallback
    if len(text) > max_line_len:
        line1 = text[:max_line_len].strip()
        line2 = text[max_line_len:].strip()
        if line1 and line2:
            return f"{line1}\n{line2}"

    return text


def export_srt_from_whisper_json(
    json_path: Path, srt_path: Path, max_line_dur: float
) -> None:
    """
    Generate SRT from Whisper JSON with robust timing.

    Core principles:
    - Trust Whisper's timing (it knows when speech starts/ends)
    - Only extend subtitles to fill small gaps (< 3s)
    - Ensure minimum reading time
    - Filter BGM phrases at the end
    """
    from settings import (
        SRT_MAX_CHARS_PER_CAPTION,
        SRT_MAX_CHARS_PER_LINE,
    )

    # Load JSON
    raw = json_path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        data = json.loads(raw.decode("utf-8", errors="ignore"))

    transcription = data.get("transcription")
    if not isinstance(transcription, list):
        raise RuntimeError("Expected 'transcription' list in Whisper JSON")

    # Extract entries
    entries = []
    for item in transcription:
        if not isinstance(item, dict):
            continue

        text = (item.get("text") or "").strip()
        if not text:
            continue

        offsets = item.get("offsets") or {}
        start_ms = offsets.get("from")
        end_ms = offsets.get("to")

        if start_ms is None or end_ms is None:
            continue

        start_s = start_ms / 1000.0
        end_s = end_ms / 1000.0

        if end_s <= start_s:
            continue

        entries.append(
            {
                "text": text,
                "start": start_s,
                "end": end_s,
            }
        )

    if not entries:
        srt_path.write_text("", encoding="utf-8")
        return

    # Find video duration for filler filtering
    total_duration = max(e["end"] for e in entries)
    filter_threshold = total_duration - 30.0

    # Process entries
    subtitles = []

    for entry in entries:
        text = entry["text"]
        start = entry["start"]
        end = entry["end"]

        # Filter BGM fillers at the end
        if start >= filter_threshold and _is_filler_phrase(text):
            continue

        # Ensure subtitle lasts long enough to read
        duration = end - start
        min_duration = _estimate_reading_time(text)

        if duration < min_duration:
            end = start + min_duration

        # Cap maximum
        if duration > max_line_dur:
            end = start + max_line_dur

        # Split if text is too long
        max_caption_chars = int(SRT_MAX_CHARS_PER_CAPTION)
        chunks = _split_text_smart(text, max_caption_chars)

        if not chunks:
            continue

        if len(chunks) == 1:
            subtitles.append(
                {
                    "start": start,
                    "end": end,
                    "text": chunks[0],
                }
            )
        else:
            # Distribute time across chunks
            total_chars = sum(len(c) for c in chunks)
            current_time = start

            for i, chunk in enumerate(chunks):
                chunk_chars = len(chunk)

                if i == len(chunks) - 1:
                    # Last chunk
                    chunk_start = current_time
                    chunk_end = end
                else:
                    # Proportional time
                    chunk_duration = (end - start) * (chunk_chars / total_chars)
                    chunk_min = _estimate_reading_time(chunk)
                    chunk_duration = max(chunk_duration, chunk_min)

                    chunk_start = current_time
                    chunk_end = min(end, chunk_start + chunk_duration)
                    current_time = chunk_end

                # Safety minimum
                if chunk_end - chunk_start < 0.3:
                    chunk_end = chunk_start + 0.3

                subtitles.append(
                    {
                        "start": chunk_start,
                        "end": chunk_end,
                        "text": chunk,
                    }
                )

    if not subtitles:
        srt_path.write_text("", encoding="utf-8")
        return

    # Sort by start time
    subtitles.sort(key=lambda x: x["start"])

    # Smart delay: if there's silence before a subtitle, start it slightly later
    # This fixes "appearing too early" without breaking sync
    for i, sub in enumerate(subtitles):
        if i == 0:
            # First subtitle - delay slightly if possible
            sub["start"] = sub["start"] + 0.15
        else:
            # Check gap from previous subtitle
            prev = subtitles[i - 1]
            gap = sub["start"] - prev["end"]

            # If there's a decent gap (> 0.5s), we can delay
            if gap > 0.5:
                # Push start later by up to 0.2s
                delay = min(0.2, gap * 0.3)  # 30% of the gap, max 0.2s
                new_start = sub["start"] + delay

                # Make sure we don’t push into the next subtitle later in the loop
                sub["start"] = new_start

    # Fill gaps between subtitles (prevents flicker during pauses)
    for i in range(len(subtitles) - 1):
        current = subtitles[i]
        next_sub = subtitles[i + 1]

        gap = next_sub["start"] - current["end"]

        if gap > 0 and gap < 3.0:
            # Small gap - extend current to meet next
            current["end"] = next_sub["start"]
        elif gap < 0:
            # Overlap - trim current
            new_end = next_sub["start"]
            if new_end - current["start"] >= 0.3:
                current["end"] = new_end
            else:
                # Keep minimum duration even if it overlaps slightly
                current["end"] = current["start"] + 0.3

    # Generate SRT
    max_line_chars = int(SRT_MAX_CHARS_PER_LINE)
    lines = []

    for idx, sub in enumerate(subtitles, start=1):
        wrapped_text = _wrap_lines(sub["text"], max_line_chars)

        lines.append(str(idx))
        lines.append(
            f"{seconds_to_srt_time(sub['start'])} --> {seconds_to_srt_time(sub['end'])}"
        )
        lines.append(wrapped_text)
        lines.append("")

    srt_path.write_text("\n".join(lines), encoding="utf-8")


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
    """Run whisper.cpp and generate SRT."""
    out_prefix = str(outbase.parent / f"{outbase.name}.{whisper_tag}")

    cmd = [
        str(whisper_bin),
        "-m",
        str(whisper_model),
        "-f",
        str(clean_wav),
        "-l",
        "ja",
        "-t",
        str(max(1, threads)),
        "-bs",
        str(max(1, beam_size)),
        "-bo",
        str(max(1, best_of)),
        "-mc",
        "3",
    ]

    if device is not None:
        cmd += ["-dev", str(device), "-fa"]

    cmd += ["-otxt", "-oj", "-of", out_prefix]

    print("[subloom] whisper cmd:", " ".join(cmd))
    run(cmd, check=True)

    json_path = Path(f"{out_prefix}.json")
    txt_path = Path(f"{out_prefix}.txt")
    srt_path = Path(f"{out_prefix}.srt")

    if not json_path.exists():
        raise RuntimeError(f"Whisper JSON not found: {json_path}")

    export_srt_from_whisper_json(json_path, srt_path, max_srt_line_dur)

    if not srt_path.exists():
        raise RuntimeError(f"Failed to generate SRT: {srt_path}")

    return WhisperPaths(
        json_path=json_path,
        srt_path=srt_path,
        txt_path=txt_path if txt_path.exists() else None,
    )
