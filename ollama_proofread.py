from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple


def _progress_bar(prefix: str, i: int, n: int) -> None:
    """Simple stderr progress bar."""
    if n <= 0:
        return
    width = 24
    done = int(width * (i / n))
    bar = "█" * done + "·" * (width - done)
    pct = int(round(100 * (i / n)))
    sys.stderr.write(f"\r{prefix} [{bar}] {i}/{n} ({pct}%)")
    sys.stderr.flush()


# =========================================================
# Settings that control how the Ollama proofreading behaves
# =========================================================


@dataclass
class OllamaConfig:
    model: str = "qwen2.5:7b-instruct"
    url: str = "http://127.0.0.1:11434/api/chat"

    # Generation behavior (keep conservative)
    temperature: float = 0.1
    top_p: float = 0.9
    num_ctx: int = 4096
    timeout_sec: int = 300

    # Chunking
    window_sec: float = 45.0
    max_chars: int = 1200

    # Optimization (reduce wasted LLM calls)
    # Only send "suspicious" lines by default
    only_suspicious: bool = True
    # If a line is mostly kanji already, likely fine
    skip_high_kanji_ratio: float = 0.55

    # Safety / behavior
    max_retries: int = 1

    # Meaning/nuance protection (LLM cannot rewrite)
    # If similarity drops too far, reject correction and keep original
    max_change_ratio: float = 0.40  # allow ~40% change; bigger => stricter
    max_abs_edit_chars: int = 28    # absolute cap on how many chars can change

    # Skip rules
    skip_music_lines: bool = True
    skip_bracket_tags: bool = True
    skip_pure_symbols: bool = True
    skip_tiny_lines: bool = True

    # Style hint: "neutral" | "anime" | "formal"
    style: str = "neutral"

    # UX
    show_progress: bool = True


# =========================================================
# Talking to Ollama
# =========================================================


def ollama_chat(cfg: OllamaConfig, messages: List[Dict[str, str]]) -> str:
    """
    Calls Ollama /api/chat and returns assistant message content as plain text.
    """
    url = cfg.url.rstrip("/")
    if not url.endswith("/api/chat"):
        # allow passing base host or /api/generate by mistake; normalize to /api/chat
        if url.endswith("/api/generate"):
            url = url[: -len("/api/generate")] + "/api/chat"
        else:
            url = url + "/api/chat"

    payload = {
        "model": cfg.model,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "num_ctx": cfg.num_ctx,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
        out = json.loads(resp.read().decode("utf-8"))

    msg = out.get("message") or {}
    return msg.get("content", "")


# =========================================================
# Prompting (locked-down, JSON only)
# =========================================================


def _style_hint(style: str) -> str:
    s = (style or "neutral").lower().strip()
    if s == "anime":
        return "Keep casual spoken Japanese if present (anime/TV dialogue vibe)."
    if s == "formal":
        return "Keep polite/formal tone if present. Do NOT make casual lines formal."
    return "Keep the original tone/register. Do NOT rewrite."


def build_proofread_prompt(lines: List[Dict[str, Any]], style: str) -> Tuple[List[Dict[str, str]], str]:
    """
    Returns (messages, input_json_string_for_debug)
    """
    sys_rules = (
        "You are a Japanese subtitle proofreader.\n"
        "Your job is ONLY to fix obvious transcription issues while preserving meaning and nuance.\n"
        f"Style: {_style_hint(style)}\n\n"
        "ABSOLUTE RULES:\n"
        "1) Do NOT paraphrase or rewrite. If a line is already fine, return it unchanged.\n"
        "2) Do NOT add/remove info. Do NOT change tone/nuance. Do NOT summarize.\n"
        "3) Do NOT merge/split lines. Keep the same number of lines.\n"
        "4) Keep each line's index i exactly the same.\n"
        "5) Allowed edits only: kanji/kana fixes, okurigana, small particle mistakes, obvious mishearing fixes,\n"
        "   punctuation/spacing normalization.\n"
        "6) Output ONLY valid JSON in EXACT schema:\n"
        '   {"lines": [{"i": <int>, "text": <string>}, ...]}\n'
        "7) No explanations. No markdown. No extra keys.\n"
    )

    examples = (
        "Acceptable examples:\n"
        "- きょう → 今日 (when clearly intended)\n"
        "- それわ → それは\n"
        "- punctuation normalize: …, 「」, ！, ？\n"
        "- obvious misheard word correction without rephrasing\n"
    )

    input_obj = {"lines": lines}
    input_json = json.dumps(input_obj, ensure_ascii=False)

    user_msg = (
        f"{examples}\n"
        "INPUT JSON:\n"
        f"{input_json}\n"
        "OUTPUT JSON ONLY:\n"
    )

    messages = [
        {"role": "system", "content": sys_rules},
        {"role": "user", "content": user_msg},
    ]
    return messages, input_json


# =========================================================
# JSON parsing + validation
# =========================================================


def _extract_json_object(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def parse_lines_json(s: str) -> List[Dict[str, Any]]:
    s = _extract_json_object(s)
    obj = json.loads(s)
    lines = obj.get("lines")
    if not isinstance(lines, list):
        raise ValueError("Missing or invalid lines[]")
    return lines


def validate_corrected(original: List[Dict[str, Any]], corrected: List[Dict[str, Any]]) -> None:
    if len(original) != len(corrected):
        raise ValueError("Line count changed")

    for o, c in zip(original, corrected):
        if o["i"] != c.get("i"):
            raise ValueError("Index mismatch")

        t = c.get("text", "")
        if not isinstance(t, str) or not t.strip():
            raise ValueError("Empty/invalid text")

        o_text = str(o.get("text", ""))
        # Basic bloat guard
        if len(t) > max(80, int(len(o_text) * 2.0)):
            raise ValueError("Over-expansion")


# =========================================================
# Meaning / nuance protections
# =========================================================


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def too_big_a_change(original: str, corrected: str, cfg: OllamaConfig) -> bool:
    o = (original or "").strip()
    c = (corrected or "").strip()
    if not o or not c:
        return True

    sim = _similarity(o, c)

    # How many characters differ (roughly)
    # (SequenceMatcher doesn't give edit distance; this is a cheap cap)
    abs_diff = abs(len(o) - len(c))
    # Also count "replacement-like" difference via similarity
    approx_changed = int(round((1.0 - sim) * max(len(o), len(c))))

    # If similarity too low, it probably rewrote
    if sim < (1.0 - cfg.max_change_ratio):
        return True

    # If change is huge even if similarity passes, reject
    if abs_diff > cfg.max_abs_edit_chars:
        return True
    if approx_changed > cfg.max_abs_edit_chars:
        return True

    return False


# =========================================================
# Chunking subtitles
# =========================================================


def chunk_by_time(subs: List[Dict[str, Any]], window_sec: float, max_chars: int) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = []
    if not subs:
        return chunks

    cur: List[Dict[str, Any]] = []
    start_t = float(subs[0]["start"])
    char_count = 0

    for s in subs:
        s_start = float(s["start"])
        s_text = str(s.get("text", ""))

        if (s_start - start_t) > window_sec or (char_count + len(s_text)) > max_chars:
            if cur:
                chunks.append(cur)
            cur = []
            start_t = s_start
            char_count = 0

        cur.append(s)
        char_count += len(s_text)

    if cur:
        chunks.append(cur)

    return chunks


# =========================================================
# Skip / filter logic (speed)
# =========================================================


_TAG_BRACKET_RE = re.compile(r"^\s*[\[\(（].*[\]\)）]\s*$")


def _count_kanji(s: str) -> int:
    return sum(1 for c in s if "\u4e00" <= c <= "\u9fff")


def _count_hiragana(s: str) -> int:
    return sum(1 for c in s if "\u3040" <= c <= "\u309f")


def should_skip_line(text: str, cfg: OllamaConfig) -> bool:
    t = (text or "").strip()
    if not t:
        return True

    if cfg.skip_tiny_lines and len(t) < 4:
        return True

    if cfg.skip_music_lines:
        if "♪" in t or "♫" in t:
            return True
        if t.lower() in ("[music]", "[bgm]", "[applause]", "[laughs]"):
            return True

    if cfg.skip_bracket_tags:
        # Short bracket tags like [SFX], (笑), （笑） etc.
        if _TAG_BRACKET_RE.match(t) and len(t) <= 14:
            return True

    if cfg.skip_pure_symbols:
        # Only punctuation / long dashes / ellipses etc.
        if re.fullmatch(r"[♪♫〜～ー…・.。！？!?、，「」『』（）\(\)\[\]\s]+", t):
            return True
        if re.fullmatch(r"[0-9０-９\s]+", t):
            return True

    # If it's already mostly kanji, it's usually already "clean"
    kanji = _count_kanji(t)
    if len(t) >= 8 and (kanji / max(1, len(t))) >= cfg.skip_high_kanji_ratio:
        return True

    return False


def looks_suspicious(text: str) -> bool:
    """
    Heuristic: lines that are hiragana-heavy tend to hide ASR/kanji errors.
    Also flags weird romaji or obvious placeholder output.
    """
    t = (text or "").strip()
    if not t:
        return False

    # romaji heavy
    if re.search(r"[A-Za-z]{4,}", t):
        return True

    hira = _count_hiragana(t)
    kanji = _count_kanji(t)

    # hiragana-heavy and not tiny
    if len(t) >= 8 and hira > kanji:
        return True

    # common ASR “weirdness”
    if "  " in t:
        return True

    return False


# =========================================================
# Proofreading chunk (safe apply)
# =========================================================


def proofread_chunk_with_ollama(cfg: OllamaConfig, payload_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages, _ = build_proofread_prompt(payload_lines, cfg.style)

    for _attempt in range(cfg.max_retries + 1):
        resp = ollama_chat(cfg, messages)
        try:
            corrected = parse_lines_json(resp)
            validate_corrected(payload_lines, corrected)

            # Meaning/nuance guard: revert any line that changes too much
            safe: List[Dict[str, Any]] = []
            for o, c in zip(payload_lines, corrected):
                o_text = str(o.get("text", ""))
                c_text = str(c.get("text", ""))

                if too_big_a_change(o_text, c_text, cfg):
                    safe.append({"i": int(o["i"]), "text": o_text})
                else:
                    safe.append({"i": int(o["i"]), "text": c_text})

            return safe

        except Exception:
            # tighten reminder without changing payload
            messages = [
                messages[0],
                {
                    "role": "user",
                    "content": (
                        messages[1]["content"]
                        + "\nREMINDER: OUTPUT JSON ONLY. NO EXTRA TEXT. DO NOT CHANGE LINE COUNT OR INDICES.\n"
                        "Do NOT paraphrase. If already fine, return the original text exactly.\n"
                    ),
                },
            ]
            time.sleep(0.10)

    # fail-soft
    return payload_lines


# =========================================================
# Entry point: what the rest of Subloom calls
# =========================================================


def ollama_proofread_subtitles(cfg: OllamaConfig, merged_subs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not merged_subs:
        return merged_subs

    chunks = chunk_by_time(merged_subs, cfg.window_sec, cfg.max_chars)
    total_chunks = len(chunks)

    if cfg.show_progress and total_chunks:
        _progress_bar("[ollama] proofreading", 0, total_chunks)

    for chunk_idx, ch in enumerate(chunks, start=1):
        payload: List[Dict[str, Any]] = []
        originals: Dict[int, str] = {}

        for s in ch:
            i = int(s["i"])
            text = str(s.get("text", ""))

            originals[i] = text

            # Skip obvious junk
            if should_skip_line(text, cfg):
                continue

            # Optional: only send lines likely to benefit
            if cfg.only_suspicious and not looks_suspicious(text):
                continue

            payload.append({"i": i, "text": text})

        if payload:
            corrected = proofread_chunk_with_ollama(cfg, payload)
            corr_map = {int(d["i"]): str(d["text"]) for d in corrected}

            # Apply corrected lines (already safety-checked)
            for s in ch:
                i = int(s["i"])
                if i in corr_map:
                    s["text"] = corr_map[i]
                else:
                    # untouched lines remain as-is
                    s["text"] = originals[i]

        if cfg.show_progress and total_chunks:
            _progress_bar("[ollama] proofreading", chunk_idx, total_chunks)

    if cfg.show_progress and total_chunks:
        _progress_bar("[ollama] proofreading", total_chunks, total_chunks)
        sys.stderr.write("\n")
        sys.stderr.flush()

    return merged_subs
