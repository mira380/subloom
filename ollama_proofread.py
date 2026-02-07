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
# (model choice, chunk size, safety rules, etc.)
# =========================================================


@dataclass
class OllamaConfig:
    model: str = "qwen2.5:7b-instruct"
    url: str = "http://127.0.0.1:11434/api/generate"

    # Generation behavior (keep conservative)
    temperature: float = 0.2
    top_p: float = 0.9
    num_ctx: int = 4096
    timeout_sec: int = 300

    # Chunking
    window_sec: float = 45.0
    max_chars: int = 1000

    # Safety / behavior
    max_retries: int = 1
    skip_music_lines: bool = True
    skip_bracket_tags: bool = True

    # Style hint: "neutral" | "anime" | "formal"
    style: str = "neutral"

    # UX
    show_progress: bool = True

    # Context chooser behavior
    context_lines: int = 1  # prev/next
    only_suspicious: bool = True

    # Nuance protection
    max_change_ratio: float = 0.40
    max_abs_edit_chars: int = 28


# =========================================================
# Talking to Ollama
# Handles the HTTP request that sends text to the LLM and
# gets corrected subtitle lines back.
# =========================================================


def ollama_generate(cfg: OllamaConfig, prompt: str) -> str:
    """
    Uses Ollama /api/chat endpoint and returns assistant message content as plain text.
    """
    url = cfg.url.rstrip("/")
    # Allow cfg.url to be either base host or full /api/chat
    if not url.endswith("/api/chat"):
        url = url + "/api/chat"

    payload = {
        "model": cfg.model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
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

    # Ollama chat response shape:
    # {"message":{"role":"assistant","content":"..."}, ...}
    msg = out.get("message") or {}
    return msg.get("content", "")


# =========================================================
# How we explain the task to the AI
# Builds the strict “proofread subtitles” instructions that
# get sent along with each chunk.
# =========================================================


def _style_hint(style: str) -> str:
    s = (style or "neutral").lower().strip()
    if s == "anime":
        return "Style: natural spoken Japanese typical of anime/TV dialogue. Keep casual tone if present.\n"
    if s == "formal":
        return "Style: keep polite/formal tone if present. Do not make casual lines formal.\n"
    return "Style: keep the original tone/register. Do not rewrite.\n"


def build_choice_prompt(items: List[Dict[str, Any]], style: str) -> str:
    rules = (
        "You are a Japanese subtitle proofreader.\n"
        "Goal: preserve meaning/nuance and choose the most correct candidate.\n"
        + _style_hint(style)
        + "\n"
        "STRICT RULES:\n"
        "1) Do NOT rewrite or paraphrase.\n"
        "2) ONLY choose from the provided candidates A/B/C.\n"
        "3) Use context (prev/next lines) to decide.\n"
        "4) If uncertain, pick A.\n"
        "5) Return ONLY valid JSON in the EXACT schema:\n"
        '   {"lines": [{"i": <int>, "pick": "A"|"B"|"C"}, ...]}\n'
        "6) Do NOT include explanations, markdown, or extra keys.\n"
    )

    input_json = json.dumps({"items": items}, ensure_ascii=False)
    return f"{rules}\nINPUT JSON:\n{input_json}\nOUTPUT JSON ONLY:\n"


# =========================================================
# Making sure the AI didn't go rogue
# Extracts JSON from the model response and checks that it
# didn’t change line counts, indices, or bloat the text.
# =========================================================


def _extract_json_object(s: str) -> str:
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def parse_choice_json(s: str) -> Dict[int, str]:
    s = _extract_json_object(s)
    obj = json.loads(s)
    lines = obj.get("lines")
    if not isinstance(lines, list):
        raise ValueError("Missing or invalid lines[]")

    picks: Dict[int, str] = {}
    for it in lines:
        if not isinstance(it, dict):
            continue
        i = it.get("i")
        pick = it.get("pick")
        if isinstance(i, int) and pick in ("A", "B", "C"):
            picks[i] = pick

    return picks


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def too_big_a_change(original: str, corrected: str, cfg: OllamaConfig) -> bool:
    o = (original or "").strip()
    c = (corrected or "").strip()
    if not o or not c:
        return True

    sim = _similarity(o, c)
    abs_diff = abs(len(o) - len(c))
    approx_changed = int(round((1.0 - sim) * max(len(o), len(c))))

    if sim < (1.0 - cfg.max_change_ratio):
        return True
    if abs_diff > cfg.max_abs_edit_chars:
        return True
    if approx_changed > cfg.max_abs_edit_chars:
        return True
    return False


# =========================================================
# Breaking subtitles into safe-sized pieces
# Sends short time windows instead of the whole file so
# context stays manageable and outputs stay consistent.
# =========================================================


def chunk_by_time(
    subs: List[Dict[str, Any]], window_sec: float, max_chars: int
) -> List[List[Dict[str, Any]]]:
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
# Lines we don't bother sending to the AI
# Skips music notes, sound tags, or tiny bracket lines that
# don't need proofreading.
# =========================================================


def should_skip_line(text: str, cfg: OllamaConfig) -> bool:
    t = (text or "").strip()
    if not t:
        return True

    if cfg.skip_music_lines:
        if "♪" in t or "♫" in t:
            return True
        if t.lower() in ("[music]", "[bgm]", "[applause]", "[laughs]"):
            return True

    if cfg.skip_bracket_tags:
        if (t.startswith("[") and t.endswith("]")) or (
            t.startswith("（") and t.endswith("）")
        ):
            if len(t) <= 12:
                return True

    return False


def _normalize_jp(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[「」『』（）()\[\]【】〈〉《》]", "", s)
    s = re.sub(r"[。、，．,.!！?？…・:：;；〜～\-—_]", "", s)
    return s


def _sim_norm(a: str, b: str) -> float:
    a = _normalize_jp(a)
    b = _normalize_jp(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _len_ratio(a: str, b: str) -> float:
    a = _normalize_jp(a)
    b = _normalize_jp(b)
    la = len(a)
    lb = len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0:
        return 9.9
    return lb / max(1, la)


def is_suspicious(
    sub: Dict[str, Any], cfg: OllamaConfig
) -> Tuple[bool, int, List[str]]:
    """Automatic gate: decide whether a line is worth sending to the LLM.

    Returns (should_check, context_lines, reasons)
    """
    cur = str(sub.get("text") or "").strip()
    whisper = str(sub.get("whisper_text") or "").strip()
    kotoba = str(sub.get("kotoba_text") or "").strip()

    reasons: List[str] = []

    # merge confidence from merge_subs (if present)
    mconf = sub.get("merge_conf")
    mdec = str(sub.get("merge_decision") or "")
    try:
        mconf_f = float(mconf) if mconf is not None else None
    except Exception:
        mconf_f = None

    if mconf_f is not None:
        if mconf_f < 0.55:
            reasons.append(f"merge_low:{mconf_f:.2f}")
        # if we kept Whisper but confidence is low, it's usually worth checking
        if mdec == "keep" and mconf_f < 0.45:
            reasons.append("kept_low")

    # obvious junk
    if re.search(r"[A-Za-z]{4,}", cur):
        reasons.append("latin_run")
    if "  " in cur:
        reasons.append("double_space")
    if "??" in cur or "？？" in cur:
        reasons.append("question_marks")

    # disagreement between engines (best signal)
    if whisper and kotoba:
        sim_wk = _sim_norm(whisper, kotoba)
        lr_wk = _len_ratio(whisper, kotoba)

        if (
            sim_wk < 0.80
            and (len(_normalize_jp(whisper)) + len(_normalize_jp(kotoba))) >= 10
        ):
            reasons.append(f"wk_disagree:{sim_wk:.2f}")
        if lr_wk < 0.60 or lr_wk > 1.80:
            reasons.append(f"wk_lenratio:{lr_wk:.2f}")

        # if current merged differs from both, it might be a weird pick
        sim_mc = _sim_norm(cur, whisper)
        sim_mk = _sim_norm(cur, kotoba)
        if sim_mc < 0.78 and sim_mk < 0.78 and len(_normalize_jp(cur)) >= 6:
            reasons.append("merged_off")

    # turbo-safe: very short lines usually don't need LLM
    if len(_normalize_jp(cur)) <= 3 and not reasons:
        return False, 0, []

    should = bool(reasons)

    # adaptive context: give more context only when it looks genuinely messy
    ctx = int(getattr(cfg, "context_lines", 1) or 1)
    if any(r.startswith("wk_disagree") for r in reasons):
        # strong disagreement => extra context helps a lot
        ctx = max(ctx, 2)
    if any(r.startswith("merge_low") for r in reasons):
        ctx = max(ctx, 2)
    if any(r.startswith("merge_low:0.") for r in reasons) and any(
        r.startswith("wk_disagree") for r in reasons
    ):
        ctx = max(ctx, 3)
    return should, ctx, reasons


def _tiny_fix_candidate(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    t = re.sub(r"\s+", " ", t)
    t = t.replace("...", "…").replace("・・", "…")
    t = t.replace("．", "。").replace("，", "、")
    t = t.replace("?", "？").replace("!", "！")
    return t


def build_candidates(sub: Dict[str, Any], cfg: OllamaConfig) -> List[Tuple[str, str]]:
    """
    Candidate sets for the chooser model.
    A: Whisper text (timing source)
    B: Current merged text with tiny safe normalization
    C: Kotoba text (alt hypothesis)

    Notes:
    - This does NOT ask the model to write new text.
    - If kotoba_text is missing, C falls back to merged.
    """
    whisper = str(sub.get("whisper_text") or sub.get("text") or "").strip()
    merged = str(sub.get("text") or "").strip()
    kotoba = str(sub.get("kotoba_text") or "").strip()

    b = _tiny_fix_candidate(merged)

    if not kotoba:
        kotoba = merged
    if not whisper:
        whisper = merged

    return [("A", whisper), ("B", b), ("C", kotoba)]


def _gather_context(lines: List[str], idx: int, k: int) -> Tuple[List[str], List[str]]:
    prev = []
    nxt = []
    for j in range(1, k + 1):
        if idx - j >= 0:
            prev.append(lines[idx - j])
        if idx + j < len(lines):
            nxt.append(lines[idx + j])
    prev.reverse()
    return prev, nxt


def choose_with_context(
    cfg: OllamaConfig,
    chunk: List[Dict[str, Any]],
) -> Dict[int, str]:
    chunk_sorted = sorted(chunk, key=lambda d: int(d["i"]))
    texts = [str(d.get("text", "")) for d in chunk_sorted]
    ids = [int(d["i"]) for d in chunk_sorted]

    items: List[Dict[str, Any]] = []
    for local_idx, (i, cur_text) in enumerate(zip(ids, texts)):
        if should_skip_line(cur_text, cfg):
            continue

        ctx_k = max(0, int(getattr(cfg, "context_lines", 1) or 1))
        if cfg.only_suspicious:
            ok, ctx_k, _reasons = is_suspicious(chunk_sorted[local_idx], cfg)
            if not ok:
                continue

        prev, nxt = _gather_context(texts, local_idx, ctx_k)

        cands = build_candidates(chunk_sorted[local_idx], cfg)

        items.append(
            {
                "i": i,
                "prev": prev,
                "cur": cur_text,
                "next": nxt,
                "candidates": [{"k": k, "text": t} for (k, t) in cands],
            }
        )

    if not items:
        return {}

    prompt = build_choice_prompt(items, cfg.style)

    for attempt in range(cfg.max_retries + 1):
        resp = ollama_generate(cfg, prompt)
        try:
            picks = parse_choice_json(resp)

            out: Dict[int, str] = {}
            for it in items:
                i = int(it["i"])
                cur_text = str(it["cur"])
                cand_map = {c["k"]: str(c["text"]) for c in it["candidates"]}

                pick = picks.get(i, "A")
                chosen = cand_map.get(pick, cur_text)

                # leash: if it picked something that changes too much, revert
                if too_big_a_change(cur_text, chosen, cfg):
                    chosen = cur_text

                # extra drift guard: chosen must stay close to at least ONE candidate
                # (this catches rare formatting / parsing weirdness)
                try:
                    sims = [
                        _sim_norm(chosen, cand_map.get("A", "")),
                        _sim_norm(chosen, cand_map.get("B", "")),
                        _sim_norm(chosen, cand_map.get("C", "")),
                    ]
                    if max(sims) < 0.85 and len(_normalize_jp(chosen)) >= 6:
                        chosen = cur_text
                except Exception:
                    pass

                out[i] = chosen

            return out
        except Exception:
            prompt = (
                build_choice_prompt(items, cfg.style)
                + "\nREMINDER: OUTPUT JSON ONLY. NO EXTRA TEXT. Picks must be A/B/C. If unsure pick A.\n"
            )
            time.sleep(0.12)

    return {}


# =========================================================
# What the rest of Subloom calls
# Entry point that runs proofreading over the subtitle list,
# chunk by chunk.
# =========================================================


def ollama_proofread_subtitles(
    cfg: OllamaConfig, merged_subs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not merged_subs:
        return merged_subs

    chunks = chunk_by_time(merged_subs, cfg.window_sec, cfg.max_chars)

    total_chunks = len(chunks)
    if cfg.show_progress and total_chunks:
        _progress_bar("[ollama] context-check", 0, total_chunks)

    for chunk_idx, ch in enumerate(chunks, start=1):
        chosen_map = choose_with_context(cfg, ch)

        if chosen_map:
            for s in ch:
                i = int(s["i"])
                if i in chosen_map:
                    s["text"] = chosen_map[i]

        if cfg.show_progress and total_chunks:
            _progress_bar("[ollama] context-check", chunk_idx, total_chunks)

    if cfg.show_progress and total_chunks:
        _progress_bar("[ollama] context-check", total_chunks, total_chunks)
        sys.stderr.write("\n")
        sys.stderr.flush()

    return merged_subs
