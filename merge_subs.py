from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional


# ----------------------------
# Merge “strictness” knobs
# Keep these here so the merge stays self-contained.
# ----------------------------

MAX_SPAN = 3

SIM_REPLACE_AT_LEAST = 0.65
MIN_OVERLAP_COVERAGE = 0.35
MIN_LEN_RATIO = 0.60
MAX_LEN_RATIO = 1.80

TOPIC_JUMP_OVERLAP_AT_LEAST = 0.55
TOPIC_JUMP_SIM_BELOW = 0.35

KOTOBA_MERGE_GAP_S = 0.25
KOTOBA_MERGE_MAX_CHARS = 60

INSERT_GAP_MIN_S = 0.55
INSERT_MAX_SEG_DUR_S = 6.5
INSERT_MIN_OVERLAP_WITH_WHISPER = 0.10
INSERT_MAX_CONSECUTIVE = 3

W_SIM = 1.30
W_COVER = 1.00
W_LEN_PENALTY = 0.35


# ----------------------------
# Low-level runner + progress
# ----------------------------


@dataclass
class RunResult:
    code: int
    stdout: str
    stderr: str


def progress_bar(prefix: str, i: int, n: int) -> None:
    """Simple stderr progress bar."""
    if n <= 0:
        return
    width = 24
    done = int(width * (i / n))
    bar = "█" * done + "·" * (width - done)
    pct = int(round(100 * (i / n)))
    sys.stderr.write(f"\r{prefix} [{bar}] {i}/{n} ({pct}%)")
    sys.stderr.flush()


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> RunResult:
    """
    Wrapper around subprocess that strips Vulkan overlay vars.
    Keeps whisper.cpp Vulkan runs from randomly breaking.
    """
    env = os.environ.copy()
    env.pop("VK_INSTANCE_LAYERS", None)
    env.pop("VK_LAYER_PATH", None)
    env["MANGOHUD"] = "0"
    env["STEAM_DISABLE_OVERLAY"] = "1"

    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if check and p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  {' '.join(cmd)}\n\n"
            f"exit={p.returncode}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}"
        )
    return RunResult(p.returncode, p.stdout, p.stderr)


# ----------------------------
# Timestamp helpers
# ----------------------------


def srt_time_to_seconds(ts: str) -> float:
    hh, mm, rest = ts.split(":")
    ss, mmm = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(mmm) / 1000.0


def seconds_to_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0.0, hi - lo)


# ----------------------------
# SRT parsing/writing
# ----------------------------


@dataclass
class SrtItem:
    idx: int
    start: str
    end: str
    text: str

    # extra fields (ignored by writer, used for Ollama context choosing)
    whisper_text: str = ""
    kotoba_text: str = ""

    # merge diagnostics (used for automatic Ollama gating + reports)
    merge_decision: str = ""  # "replace" | "keep" | "insert"
    merge_sim: float = 0.0
    merge_cov: float = 0.0
    merge_lr: float = 0.0
    merge_score: float = 0.0
    merge_conf: float = 0.0


@dataclass
class Seg:
    i: int
    t0: float
    t1: float
    text: str


def parse_srt(path: Path) -> list[SrtItem]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", raw.strip(), flags=re.M)
    items: list[SrtItem] = []
    for blk in blocks:
        lines = [ln.rstrip("\r") for ln in blk.splitlines() if ln.strip() != ""]
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        m = re.match(
            r"(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)", lines[1]
        )
        if not m:
            continue
        start, end = m.group(1), m.group(2)
        text = "\n".join(lines[2:]).strip()
        items.append(SrtItem(idx=idx, start=start, end=end, text=text))
    return items


def write_srt_items(items: list[SrtItem], out_path: Path) -> None:
    lines: list[str] = []
    for idx, it in enumerate(items, 1):
        lines.append(str(idx))
        lines.append(f"{it.start} --> {it.end}")
        lines.append((it.text or "").strip())
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def polish_timing(
    items: list[SrtItem],
    *,
    max_dur_s: float = 4.0,
    min_gap_s: float = 0.06,
    short_len_n: int = 3,
    short_start_nudge_s: float = 0.12,
    max_hold_per_char_s: float = 0.12,
    min_dur_s: float = 0.65,
) -> list[SrtItem]:
    """
    Timing stabilizer for "early" + "weird holds" cases.

    What it does:
    - Caps ridiculously long holds (esp. short lines like 「失礼します」)
    - Nudges very short interjections slightly later (reduces "too early" feel)
    - Enforces no overlaps + small minimum gap
    - Splits lines that are super long in time (simple midpoint split, text unchanged)
    """
    if not items:
        return items

    items = sorted(items, key=lambda x: srt_time_to_seconds(x.start))

    out: list[SrtItem] = []
    for it in items:
        t0 = srt_time_to_seconds(it.start)
        t1 = srt_time_to_seconds(it.end)
        if t1 <= t0:
            t1 = t0 + min_dur_s

        # 1) Cap long holds based on text length (prevents 10s "失礼します" etc)
        ntext = normalize_jp(it.text)
        nlen = len(ntext)
        if nlen > 0:
            cap = max(min_dur_s, min(max_dur_s, max_hold_per_char_s * nlen))
            if (t1 - t0) > cap:
                t1 = t0 + cap

        # 2) Nudge super short lines later (anime "はあ", "え", "付け", etc)
        if nlen > 0 and nlen <= short_len_n:
            # don't nudge if it's already extremely short
            if (t1 - t0) > 0.20:
                t0 = min(t0 + short_start_nudge_s, t1 - 0.20)

        # 3) Split lines that are still too long in time (rare after cap, but helps)
        dur = t1 - t0
        if dur > max_dur_s + 0.25:
            mid = t0 + dur * 0.5
            a = SrtItem(
                idx=it.idx,
                start=seconds_to_srt_time(t0),
                end=seconds_to_srt_time(mid),
                text=it.text,
                whisper_text=getattr(it, "whisper_text", ""),
                kotoba_text=getattr(it, "kotoba_text", ""),
                merge_decision=getattr(it, "merge_decision", ""),
                merge_sim=getattr(it, "merge_sim", 0.0),
                merge_cov=getattr(it, "merge_cov", 0.0),
                merge_lr=getattr(it, "merge_lr", 0.0),
                merge_score=getattr(it, "merge_score", 0.0),
                merge_conf=getattr(it, "merge_conf", 0.0),
            )
            b = SrtItem(
                idx=it.idx,
                start=seconds_to_srt_time(mid + min_gap_s),
                end=seconds_to_srt_time(t1),
                text=it.text,
                whisper_text=getattr(it, "whisper_text", ""),
                kotoba_text=getattr(it, "kotoba_text", ""),
                merge_decision=getattr(it, "merge_decision", ""),
                merge_sim=getattr(it, "merge_sim", 0.0),
                merge_cov=getattr(it, "merge_cov", 0.0),
                merge_lr=getattr(it, "merge_lr", 0.0),
                merge_score=getattr(it, "merge_score", 0.0),
                merge_conf=getattr(it, "merge_conf", 0.0),
            )
            out.append(a)
            out.append(b)
        else:
            it.start = seconds_to_srt_time(t0)
            it.end = seconds_to_srt_time(t1)
            out.append(it)

    # 4) Enforce monotonic timing + minimum gaps (no overlap)
    out.sort(key=lambda x: srt_time_to_seconds(x.start))
    for i in range(1, len(out)):
        prev = out[i - 1]
        cur = out[i]

        p0 = srt_time_to_seconds(prev.start)
        p1 = srt_time_to_seconds(prev.end)
        c0 = srt_time_to_seconds(cur.start)
        c1 = srt_time_to_seconds(cur.end)

        # ensure prev has sane duration
        if p1 <= p0:
            p1 = p0 + min_dur_s
            prev.end = seconds_to_srt_time(p1)

        # push current start forward if it overlaps or is too tight
        need = p1 + min_gap_s
        if c0 < need:
            shift = need - c0
            c0 = need
            # keep end from going backwards; preserve duration if possible
            c1 = max(c0 + 0.20, c1 + shift)

        # final sanity
        if c1 <= c0:
            c1 = c0 + min_dur_s

        cur.start = seconds_to_srt_time(c0)
        cur.end = seconds_to_srt_time(c1)

    return out


def dedupe_consecutive_items(
    items: list[SrtItem],
    *,
    window_s: float = 1.5,
    short_len: int = 10,
    sim_at_least: float = 0.92,
    min_overlap_ratio: float = 0.65,
) -> list[SrtItem]:
    """
    Collapse consecutive duplicates / near-duplicates.

    Tier A (always):
      - exact normalized match (no length limit), within window_s

    Tier B (safe):
      - high similarity AND strong time overlap
      - more permissive for short lines
    """
    if not items:
        return items

    def _overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
        inter = overlap(a0, a1, b0, b1)
        denom = max(1e-6, min(a1 - a0, b1 - b0))
        return inter / denom

    out: list[SrtItem] = []
    for cur in items:
        if not out:
            out.append(cur)
            continue

        prev = out[-1]

        p0 = srt_time_to_seconds(prev.start)
        p1 = srt_time_to_seconds(prev.end)
        c0 = srt_time_to_seconds(cur.start)
        c1 = srt_time_to_seconds(cur.end)

        gap = c0 - p1
        if gap > window_s:
            out.append(cur)
            continue

        pn = normalize_jp(prev.text)
        cn = normalize_jp(cur.text)

        if not pn or not cn:
            out.append(cur)
            continue

        # Tier A: exact normalized duplicate (no length limit)
        if pn == cn:
            prev.end = max(prev.end, cur.end, key=srt_time_to_seconds)
            continue

        # Tier B: near-duplicate, but require strong overlap + high similarity
        ov = _overlap_ratio(p0, p1, c0, c1)
        sim = sim_ratio_norm(pn, cn)

        is_short = (len(pn) <= short_len) and (len(cn) <= short_len)

        # short lines are allowed a bit more easily
        need_ov = min_overlap_ratio if not is_short else (min_overlap_ratio * 0.5)

        if sim >= sim_at_least and ov >= need_ov:
            prev.end = max(prev.end, cur.end, key=srt_time_to_seconds)
            # keep prev.text (don’t flip-flop between minor variants)
            continue

        out.append(cur)

    return out


# ----------------------------
# Optional Ollama proofreading (keeps timings)
# ----------------------------


def apply_ollama_to_srt_items(items: list[SrtItem], cfg) -> list[SrtItem]:
    """
    cfg is an OllamaConfig from ollama_proofread.py
    (kept untyped here so merge_subs doesn't need to import dataclasses from there)
    """
    from ollama_proofread import ollama_proofread_subtitles

    merged_subs: list[dict] = []
    for it in items:
        merged_subs.append(
            {
                "i": int(it.idx),
                "start": srt_time_to_seconds(it.start),
                "end": srt_time_to_seconds(it.end),
                "text": it.text or "",
                "whisper_text": (getattr(it, "whisper_text", "") or ""),
                "kotoba_text": (getattr(it, "kotoba_text", "") or ""),
                "merge_decision": (getattr(it, "merge_decision", "") or ""),
                "merge_sim": float(getattr(it, "merge_sim", 0.0) or 0.0),
                "merge_cov": float(getattr(it, "merge_cov", 0.0) or 0.0),
                "merge_lr": float(getattr(it, "merge_lr", 0.0) or 0.0),
                "merge_score": float(getattr(it, "merge_score", 0.0) or 0.0),
                "merge_conf": float(getattr(it, "merge_conf", 0.0) or 0.0),
            }
        )

    merged_subs = ollama_proofread_subtitles(cfg, merged_subs)

    text_map = {int(d["i"]): d["text"] for d in merged_subs}
    for it in items:
        it.text = text_map.get(int(it.idx), it.text)

    return items


# ----------------------------
# Normalization + similarity
# ----------------------------

_RE_SPACES = re.compile(r"\s+")
_RE_BRACKETS = re.compile(r"[「」『』（）()\[\]【】〈〉《》]")
_RE_PUNCT = re.compile(r"[。、，．,.!！?？…・:：;；〜～\-—_]")


def normalize_jp(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = _RE_SPACES.sub("", s)
    s = _RE_BRACKETS.sub("", s)
    s = _RE_PUNCT.sub("", s)
    return s


_RE_JP_SPACE = re.compile(r"[ \t]+")

# sentence-ish endings where "。" is usually safe if missing
_RE_END_OK = re.compile(
    r"(だ|です|ます|だった|でした|だろ|だろう|よ|ね|な|ぞ|か|かな|かも|じゃん|じゃない|だな|かよ)$"
)

# connectors where a comma is often helpful
_RE_COMMA_CONNECT = re.compile(
    r"(けど|けれど|から|ので|のに|のだが|のですが|けどさ|でも|しかし|そして|それで|だから|つまり)"
)

# avoid adding punctuation if line already ends with these
_END_PUNCT = ("。", "！", "？", "…", "!", "?", "）", ")", "」", "』")


def punctuate_jp_line(s: str) -> str:
    """
    Light punctuation for JP subtitle lines.
    Goal: readability, not perfect grammar.
    """
    if not s:
        return s

    # normalize spaces (common from ASR)
    s = s.replace("\u3000", " ")
    s = _RE_JP_SPACE.sub(" ", s).strip()

    # split multi-line blocks and punctuate each line gently
    parts = [p.strip() for p in s.split("\n")]
    out_parts: list[str] = []

    for line in parts:
        if not line:
            continue

        # don't mess with obvious SFX/music bracket lines
        if line.startswith(("♪", "♬")) or (
            line.startswith("（") and line.endswith("）")
        ):
            out_parts.append(line)
            continue
        if line.startswith("[") and line.endswith("]"):
            out_parts.append(line)
            continue

        # add commas after common connectors if missing
        # (only if it looks like a longer clause)
        if len(normalize_jp(line)) >= 10:
            line = _RE_COMMA_CONNECT.sub(r"\1、", line)

        # collapse accidental double commas
        line = line.replace("、、", "、")

        # add sentence end punctuation if it looks like it needs it
        if not line.endswith(_END_PUNCT):
            # question-ish
            if (
                line.endswith(("か", "の", "かな", "だろ", "だろう"))
                and len(normalize_jp(line)) >= 4
            ):
                line += "？"
            # exclamation-ish
            elif line.endswith(("！", "!")):
                pass
            # statement-ish
            elif _RE_END_OK.search(line) and len(normalize_jp(line)) >= 4:
                line += "。"

        out_parts.append(line)

    return "\n".join(out_parts).strip()


def apply_punctuation(items: list[SrtItem]) -> list[SrtItem]:
    for it in items:
        it.text = punctuate_jp_line(it.text or "")
    return items


def sim_ratio_norm(a_norm: str, b_norm: str) -> float:
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def load_whispercpp_segments(json_path: Path) -> list[Seg]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    tr = data.get("transcription")
    if not isinstance(tr, list):
        keys = (
            ", ".join(list(data.keys())[:30])
            if isinstance(data, dict)
            else "<not a dict>"
        )
        raise RuntimeError(
            f"Expected 'transcription' list in JSON: {json_path}. Keys: {keys}"
        )

    segs: list[Seg] = []
    for item in tr:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        off = item.get("offsets") or {}
        a = off.get("from")
        b = off.get("to")
        if a is None or b is None:
            continue
        t0 = float(a) / 1000.0
        t1 = float(b) / 1000.0
        if t1 <= t0:
            t1 = t0 + 0.5
        segs.append(Seg(i=len(segs), t0=t0, t1=t1, text=text))
    return segs


def merge_kotoba_micro_segments(segs: list[Seg]) -> list[Seg]:
    """
    Merge tiny Kotoba micro-segments into bigger chunks.
    Optimization: normalize once per segment and track normalized length.
    """
    if not segs:
        return segs

    norm = [normalize_jp(s.text) for s in segs]
    norm_len = [len(x) for x in norm]

    merged: list[Seg] = []
    cur = Seg(i=0, t0=segs[0].t0, t1=segs[0].t1, text=segs[0].text)
    cur_norm_len = norm_len[0]

    for idx, s in enumerate(segs[1:], start=1):
        gap = s.t0 - cur.t1
        if gap <= KOTOBA_MERGE_GAP_S and (
            cur_norm_len + norm_len[idx] <= KOTOBA_MERGE_MAX_CHARS
        ):
            cur = Seg(i=0, t0=cur.t0, t1=max(cur.t1, s.t1), text=(cur.text + s.text))
            cur_norm_len += norm_len[idx]
        else:
            merged.append(cur)
            cur = Seg(i=0, t0=s.t0, t1=s.t1, text=s.text)
            cur_norm_len = norm_len[idx]

    merged.append(cur)

    out: list[Seg] = []
    for i, s in enumerate(merged):
        out.append(Seg(i=i, t0=s.t0, t1=s.t1, text=s.text.strip()))
    return out


# ----------------------------
# Smart merge brain
# ----------------------------


def span_overlap_coverage(
    w0: float, w1: float, segs: list[Seg], start: int, span: int
) -> float:
    win = max(1e-6, (w1 - w0))
    ov = 0.0
    for s in segs[start : start + span]:
        ov += overlap(w0, w1, s.t0, s.t1)
    return min(1.0, ov / win)


def best_span_for_whisper_line(
    w_text: str,
    w0: float,
    w1: float,
    segs: list[Seg],
    segs_norm: list[str],
    segs_norm_len: list[int],
    k_start_hint: int,
) -> tuple[Optional[tuple[int, int]], float, float, float, float, str]:
    """
    Optimized version:
    - normalize Whisper once
    - reuse pre-normalized Kotoba segments
    - build span normalized text incrementally per start
    - avoid repeated normalize_jp()/SequenceMatcher calls
    """
    if not segs:
        return None, -1e9, 0.0, 0.0, 0.0, ""

    w_norm = normalize_jp(w_text)
    w_len = len(w_norm)

    k = max(0, min(k_start_hint, len(segs)))
    while k < len(segs) and segs[k].t1 < w0:
        k += 1

    start_min = max(0, k - 2)
    start_max = min(len(segs) - 1, k + 6)

    best: Optional[tuple[int, int]] = None
    best_score = -1e9
    best_sim = 0.0
    best_cov = 0.0
    best_lr = 0.0
    best_txt = ""

    for sidx in range(start_min, start_max + 1):
        if segs[sidx].t0 > w1:
            break

        raw_acc = ""
        norm_acc = ""
        norm_len_acc = 0

        # build spans 1..MAX_SPAN incrementally (same start)
        for span in range(1, MAX_SPAN + 1):
            j = sidx + span - 1
            if j >= len(segs):
                break

            raw_acc += segs[j].text
            norm_acc += segs_norm[j]
            norm_len_acc += segs_norm_len[j]

            txt = raw_acc.strip()
            if not txt:
                continue

            cov = span_overlap_coverage(w0, w1, segs, sidx, span)
            if cov <= 0.0:
                continue

            sim = sim_ratio_norm(w_norm, norm_acc)

            # length ratio using normalized lengths (no repeated normalize)
            if w_len == 0 and norm_len_acc == 0:
                lr = 1.0
            elif w_len == 0:
                lr = 9.9
            else:
                lr = max(0.0, norm_len_acc / max(1, w_len))

            lp = abs(math.log(lr)) if lr > 0 else 2.0
            score = (W_SIM * sim) + (W_COVER * cov) - (W_LEN_PENALTY * lp)

            if score > best_score:
                best_score = score
                best = (sidx, span)
                best_sim = sim
                best_cov = cov
                best_lr = lr
                best_txt = txt

    return best, best_score, best_sim, best_cov, best_lr, best_txt


def _clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(x)


def _strip_leading_noise(s: str) -> str:
    return (s or "").lstrip(" 　\t\n「『（()[]【】」』）.,。、!！?？ー-")


def _prefix_noise_penalty(whisper: str, cand: str) -> float:
    """
    Penalize candidates that look like 'whisper' but with 1-3 extra leading chars.
    Example: 'しかべてのものを映す鏡' vs '全てのものを映す鏡'
    """
    w = _strip_leading_noise(whisper)
    c = _strip_leading_noise(cand)
    if not w or not c:
        return 0.0

    # If same starting char, it's usually fine.
    if c[0] == w[0]:
        return 0.0

    # If whisper's first char appears shortly after candidate start, candidate likely has junk prefix.
    j = c.find(w[0])
    if 0 < j <= 3:
        return 0.50 + 0.20 * j  # strong penalty
    return 0.0


def _confidence_from_match(
    sim: float, cov: float, lr: float, *, topic_jump: bool
) -> float:
    """Heuristic confidence for a Whisper<->Kotoba match.

    0.0 = likely wrong / needs help
    1.0 = very safe
    """
    if topic_jump:
        return 0.0

    base = 0.55 * sim + 0.45 * cov

    # if lengths are way off, treat as suspicious
    if lr < MIN_LEN_RATIO or lr > MAX_LEN_RATIO:
        base *= 0.55

    return _clamp01(base)


def _write_lowconf_txt(path: Path, rows: list[dict], thresh: float) -> None:
    lows = [r for r in rows if float(r.get("conf", 0.0)) < thresh]
    if not lows:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass
        return

    out: list[str] = []
    out.append(f"LOW CONFIDENCE (< {thresh:.2f})")
    out.append("")
    for r in lows:
        out.append(
            f"[{r.get('decision', '?').upper()} conf={float(r.get('conf', 0.0)):.2f}] {r.get('start', '')} --> {r.get('end', '')}"
        )
        w = (r.get("whisper") or "").strip()
        k = (r.get("kotoba") or "").strip()
        c = (r.get("chosen") or "").strip()
        if w:
            out.append(f"  whisper: {w}")
        if k:
            out.append(f"  kotoba : {k}")
        if c:
            out.append(f"  final  : {c}")
        reasons = r.get("reasons") or []
        if reasons:
            out.append(f"  reasons: {', '.join(reasons)}")
        out.append("")
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


# ----------------------------
# Main merge
# ----------------------------


def auto_merge_and_insert(
    whisper_srt: Path,
    kotoba_json: Path,
    out_final_srt: Path,
    out_compare_log: Optional[Path] = None,
    ollama_cfg: Optional[object] = None,
):
    """
    Uses Whisper SRT as timing base.
    Uses Kotoba JSON segments as text suggestions + gap insertion.
    Optionally runs Ollama proofreading at the end (text only, no timing changes).
    """
    w_items = parse_srt(whisper_srt)
    w_windows = [
        (srt_time_to_seconds(it.start), srt_time_to_seconds(it.end)) for it in w_items
    ]

    k_segs_raw = load_whispercpp_segments(kotoba_json)
    k_segs = merge_kotoba_micro_segments(k_segs_raw)

    # precompute normalized kotoba segs once (big win)
    k_norm = [normalize_jp(s.text) for s in k_segs]
    k_norm_len = [len(x) for x in k_norm]

    used_k: set[int] = set()
    log: list[str] = []
    rows: list[dict] = []  # merge diagnostics for reports

    if out_compare_log:
        log.append("AUTO MERGE + GAP INSERT LOG")
        log.append(f"whisper_srt: {whisper_srt}")
        log.append(f"kotoba_json: {kotoba_json}")
        log.append(f"SIM_REPLACE_AT_LEAST={SIM_REPLACE_AT_LEAST}")
        log.append(f"MIN_OVERLAP_COVERAGE={MIN_OVERLAP_COVERAGE}")
        log.append(f"LEN_RATIO in [{MIN_LEN_RATIO}, {MAX_LEN_RATIO}]")
        log.append(
            f"TOPIC_JUMP: overlap>={TOPIC_JUMP_OVERLAP_AT_LEAST} and sim<{TOPIC_JUMP_SIM_BELOW} => block"
        )
        log.append(
            f"GAP_INSERT: gap>={INSERT_GAP_MIN_S}s, max_seg_dur={INSERT_MAX_SEG_DUR_S}s"
        )
        log.append("")

    final: list[SrtItem] = []
    k_hint = 0

    # 1) Replace text using best-span selection + guardrails (optimized)
    for it in w_items:
        w0 = srt_time_to_seconds(it.start)
        w1 = srt_time_to_seconds(it.end)

        best, score, sim, cov, lr, ktxt = best_span_for_whisper_line(
            w_text=it.text,
            w0=w0,
            w1=w1,
            segs=k_segs,
            segs_norm=k_norm,
            segs_norm_len=k_norm_len,
            k_start_hint=k_hint,
        )

        chosen = it.text
        decision = "keep"
        reasons: list[str] = []
        conf = 0.18

        if best is not None and ktxt:
            sidx, span = best
            k_hint = max(k_hint, sidx)

            topic_jump = (
                cov >= TOPIC_JUMP_OVERLAP_AT_LEAST and sim < TOPIC_JUMP_SIM_BELOW
            )

            prefix_pen = _prefix_noise_penalty(it.text, ktxt)
            prefix_bad = prefix_pen >= 0.50
            if prefix_bad:
                reasons.append("prefix_noise")

            ok = (
                (not topic_jump)
                and (not prefix_bad)
                and cov >= MIN_OVERLAP_COVERAGE
                and sim >= SIM_REPLACE_AT_LEAST
                and (MIN_LEN_RATIO <= lr <= MAX_LEN_RATIO)
            )

            if ok:
                chosen = ktxt
                decision = "replace"
                conf = _confidence_from_match(sim, cov, lr, topic_jump=topic_jump)
                for j in range(sidx, sidx + span):
                    used_k.add(j)

                if out_compare_log and normalize_jp(chosen) != normalize_jp(it.text):
                    log.append(
                        f"[REPLACE sim={sim:.2f} cov={cov:.2f} lr={lr:.2f} score={score:.2f}] {it.start} --> {it.end}"
                    )
                    log.append(f"  whisper: {it.text}")
                    log.append(f"  kotoba : {ktxt}")
                    log.append("")
            else:
                conf = (
                    _confidence_from_match(sim, cov, lr, topic_jump=topic_jump) * 0.85
                )
                reason_bits: list[str] = []
                if prefix_bad:
                    reason_bits.append("prefix_noise")
                if topic_jump:
                    reason_bits.append("topic_jump")
                if cov < MIN_OVERLAP_COVERAGE:
                    reason_bits.append(f"cov<{MIN_OVERLAP_COVERAGE}")
                if sim < SIM_REPLACE_AT_LEAST:
                    reason_bits.append(f"sim<{SIM_REPLACE_AT_LEAST}")
                if lr < MIN_LEN_RATIO or lr > MAX_LEN_RATIO:
                    reason_bits.append(f"lr not in [{MIN_LEN_RATIO},{MAX_LEN_RATIO}]")
                reasons = reason_bits

                if out_compare_log:
                    reason = ", ".join(reason_bits) if reason_bits else "guard"
                    log.append(
                        f"[KEEP {reason} sim={sim:.2f} cov={cov:.2f} lr={lr:.2f}] {it.start} --> {it.end}"
                    )
                    log.append(f"  whisper: {it.text}")
                    log.append(f"  kotoba : {ktxt}")
                    log.append("")

        final.append(
            SrtItem(
                idx=it.idx,
                start=it.start,
                end=it.end,
                text=chosen,
                whisper_text=(it.text or "").strip(),
                kotoba_text=(ktxt or "").strip(),
                merge_decision=decision,
                merge_sim=float(sim),
                merge_cov=float(cov),
                merge_lr=float(lr),
                merge_score=float(score),
                merge_conf=float(conf),
            )
        )

        rows.append(
            {
                "i": int(it.idx),
                "start": it.start,
                "end": it.end,
                "decision": decision,
                "conf": float(conf),
                "sim": float(sim),
                "cov": float(cov),
                "lr": float(lr),
                "score": float(score),
                "whisper": (it.text or "").strip(),
                "kotoba": (ktxt or "").strip(),
                "chosen": (chosen or "").strip(),
                "reasons": reasons,
            }
        )

    # 2) Gap insertion using unused Kotoba segments (optimized: linear scan + neighbor overlap only)
    inserts: list[SrtItem] = []

    k_ptr = 0
    n_k = len(k_segs)

    for i in range(len(w_items) - 1):
        a_end = srt_time_to_seconds(w_items[i].end)
        b_start = srt_time_to_seconds(w_items[i + 1].start)
        gap = b_start - a_end
        if gap < INSERT_GAP_MIN_S:
            continue

        # move pointer forward to first segment that could fit after a_end
        while k_ptr < n_k and k_segs[k_ptr].t1 <= a_end:
            k_ptr += 1

        inserted_here = 0
        j = k_ptr

        prev_w0, prev_w1 = w_windows[i]
        next_w0, next_w1 = w_windows[i + 1]

        while j < n_k and k_segs[j].t0 < b_start:
            if inserted_here >= INSERT_MAX_CONSECUTIVE:
                break

            ks = k_segs[j]

            if ks.i in used_k:
                j += 1
                continue
            if ks.t0 < a_end or ks.t1 > b_start:
                j += 1
                continue
            if (ks.t1 - ks.t0) > INSERT_MAX_SEG_DUR_S:
                j += 1
                continue

            txt = ks.text.strip()
            if not txt:
                j += 1
                continue

            # In a gap, only the neighboring whisper windows can overlap meaningfully
            if (
                overlap(ks.t0, ks.t1, prev_w0, prev_w1)
                > INSERT_MIN_OVERLAP_WITH_WHISPER
                or overlap(ks.t0, ks.t1, next_w0, next_w1)
                > INSERT_MIN_OVERLAP_WITH_WHISPER
            ):
                j += 1
                continue

            inserts.append(
                SrtItem(
                    idx=0,
                    start=seconds_to_srt_time(ks.t0),
                    end=seconds_to_srt_time(ks.t1),
                    text=txt,
                    whisper_text=txt,
                    kotoba_text=txt,
                    merge_decision="insert",
                    merge_sim=0.0,
                    merge_cov=0.0,
                    merge_lr=1.0,
                    merge_score=0.0,
                    merge_conf=0.45,
                )
            )

            rows.append(
                {
                    "i": 0,
                    "start": seconds_to_srt_time(ks.t0),
                    "end": seconds_to_srt_time(ks.t1),
                    "decision": "insert",
                    "conf": 0.45,
                    "sim": 0.0,
                    "cov": 0.0,
                    "lr": 1.0,
                    "score": 0.0,
                    "whisper": "",
                    "kotoba": txt,
                    "chosen": txt,
                    "reasons": ["gap_insert"],
                }
            )

            used_k.add(ks.i)
            inserted_here += 1

            if out_compare_log:
                log.append(
                    f"[INSERT] {seconds_to_srt_time(ks.t0)} --> {seconds_to_srt_time(ks.t1)}"
                )
                log.append(f"  kotoba: {txt}")
                log.append("")

            j += 1

    combined = final + inserts
    combined.sort(key=lambda x: srt_time_to_seconds(x.start))

    # ---- Ollama proofreading (optional) ----
    if ollama_cfg is not None:
        print(
            f"[subloom] Ollama proofreading: model={getattr(ollama_cfg, 'model', '?')}, style={getattr(ollama_cfg, 'style', '?')}"
        )

        # debug: count how many lines actually change
        before = [it.text for it in combined]

        combined = apply_ollama_to_srt_items(combined, ollama_cfg)

        after = [it.text for it in combined]
        changed = sum(1 for a, b in zip(before, after) if a != b)

        print(f"[subloom] Ollama changed {changed}/{len(combined)} lines")

    # ---- automatic merge report (always) ----
    # Keep it lightweight: a JSON stats file + a tiny low-confidence text file.
    try:
        report_json = out_final_srt.with_name(out_final_srt.stem + ".report.json")
        lowconf_txt = out_final_srt.with_name(out_final_srt.stem + ".lowconf.txt")
        thresh = 0.55

        report_json.write_text(
            json.dumps(
                {
                    "whisper_srt": str(whisper_srt),
                    "kotoba_json": str(kotoba_json),
                    "final_srt": str(out_final_srt),
                    "lowconf_thresh": thresh,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_lowconf_txt(lowconf_txt, rows, thresh)
    except Exception:
        # reports are best-effort; never fail the run because of them
        pass

    # collapse obvious consecutive duplicates (anime-style spam repeats)
    combined = dedupe_consecutive_items(combined, window_s=1.5, short_len=10)
    combined = polish_timing(combined)
    write_srt_items(combined, out_final_srt)

    if out_compare_log:
        out_compare_log.write_text("\n".join(log).rstrip() + "\n", encoding="utf-8")
