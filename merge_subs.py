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
# Low-level runner + progressruforts.
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


def normalize_jp(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[「」『』（）()\[\]【】〈〉《》]", "", s)
    s = re.sub(r"[。、，．,.!！?？…・:：;；〜～\-—_]", "", s)
    return s


def sim_ratio(a: str, b: str) -> float:
    a2 = normalize_jp(a)
    b2 = normalize_jp(b)
    if not a2 and not b2:
        return 1.0
    if not a2 or not b2:
        return 0.0
    return SequenceMatcher(None, a2, b2).ratio()


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
    if not segs:
        return segs
    merged: list[Seg] = []
    cur = Seg(i=0, t0=segs[0].t0, t1=segs[0].t1, text=segs[0].text)
    for s in segs[1:]:
        gap = s.t0 - cur.t1
        if gap <= KOTOBA_MERGE_GAP_S and (
            len(normalize_jp(cur.text)) + len(normalize_jp(s.text))
            <= KOTOBA_MERGE_MAX_CHARS
        ):
            cur = Seg(i=0, t0=cur.t0, t1=max(cur.t1, s.t1), text=(cur.text + s.text))
        else:
            merged.append(cur)
            cur = Seg(i=0, t0=s.t0, t1=s.t1, text=s.text)
    merged.append(cur)

    out: list[Seg] = []
    for idx, s in enumerate(merged):
        out.append(Seg(i=idx, t0=s.t0, t1=s.t1, text=s.text.strip()))
    return out


# ----------------------------
# Smart merge brain
# ----------------------------


def len_ratio(a: str, b: str) -> float:
    a2 = normalize_jp(a)
    b2 = normalize_jp(b)
    if not a2 and not b2:
        return 1.0
    if not a2:
        return 9.9
    return max(0.0, len(b2) / max(1, len(a2)))


def span_text(segs: list[Seg], start: int, span: int) -> str:
    return "".join(s.text for s in segs[start : start + span]).strip()


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
    k_start_hint: int,
) -> tuple[Optional[tuple[int, int]], float, float, float, float, str]:
    if not segs:
        return None, -1e9, 0.0, 0.0, 0.0, ""

    k = max(0, min(k_start_hint, len(segs)))
    while k < len(segs) and segs[k].t1 < w0:
        k += 1

    start_min = max(0, k - 2)
    start_max = min(len(segs) - 1, k + 6)

    best = None
    best_score = -1e9
    best_sim = 0.0
    best_cov = 0.0
    best_lr = 0.0
    best_txt = ""

    for sidx in range(start_min, start_max + 1):
        if segs[sidx].t0 > w1:
            break

        for span in range(1, MAX_SPAN + 1):
            if sidx + span > len(segs):
                continue

            txt = span_text(segs, sidx, span)
            if not txt:
                continue

            cov = span_overlap_coverage(w0, w1, segs, sidx, span)
            if cov <= 0.0:
                continue

            sim = sim_ratio(w_text, txt)
            lr = len_ratio(w_text, txt)

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


def overlaps_any_whisper(
    seg: Seg, whisper_windows: list[tuple[float, float]], max_overlap: float
) -> bool:
    for w0, w1 in whisper_windows:
        if overlap(seg.t0, seg.t1, w0, w1) > max_overlap:
            return True
    return False


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

    used_k: set[int] = set()
    log: list[str] = []

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

    # 1) Replace text using best-span selection + guardrails
    for it in w_items:
        w0 = srt_time_to_seconds(it.start)
        w1 = srt_time_to_seconds(it.end)

        best, score, sim, cov, lr, ktxt = best_span_for_whisper_line(
            w_text=it.text,
            w0=w0,
            w1=w1,
            segs=k_segs,
            k_start_hint=k_hint,
        )

        chosen = it.text

        if best is not None and ktxt:
            sidx, span = best
            k_hint = max(k_hint, sidx)

            topic_jump = (
                cov >= TOPIC_JUMP_OVERLAP_AT_LEAST and sim < TOPIC_JUMP_SIM_BELOW
            )

            ok = (
                (not topic_jump)
                and cov >= MIN_OVERLAP_COVERAGE
                and sim >= SIM_REPLACE_AT_LEAST
                and (lr >= MIN_LEN_RATIO and lr <= MAX_LEN_RATIO)
            )

            if ok:
                chosen = ktxt
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
                if out_compare_log:
                    reason_bits = []
                    if topic_jump:
                        reason_bits.append("topic_jump")
                    if cov < MIN_OVERLAP_COVERAGE:
                        reason_bits.append(f"cov<{MIN_OVERLAP_COVERAGE}")
                    if sim < SIM_REPLACE_AT_LEAST:
                        reason_bits.append(f"sim<{SIM_REPLACE_AT_LEAST}")
                    if lr < MIN_LEN_RATIO or lr > MAX_LEN_RATIO:
                        reason_bits.append(
                            f"lr not in [{MIN_LEN_RATIO},{MAX_LEN_RATIO}]"
                        )
                    reason = ", ".join(reason_bits) if reason_bits else "guard"
                    log.append(
                        f"[KEEP {reason} sim={sim:.2f} cov={cov:.2f} lr={lr:.2f}] {it.start} --> {it.end}"
                    )
                    log.append(f"  whisper: {it.text}")
                    log.append(f"  kotoba : {ktxt}")
                    log.append("")

        final.append(SrtItem(idx=it.idx, start=it.start, end=it.end, text=chosen))

    # 2) Gap insertion using unused Kotoba segments
    inserts: list[SrtItem] = []
    for i in range(len(w_items) - 1):
        a_end = srt_time_to_seconds(w_items[i].end)
        b_start = srt_time_to_seconds(w_items[i + 1].start)
        gap = b_start - a_end
        if gap < INSERT_GAP_MIN_S:
            continue

        inserted_here = 0
        for ks in k_segs:
            if inserted_here >= INSERT_MAX_CONSECUTIVE:
                break
            if ks.i in used_k:
                continue
            if ks.t0 < a_end or ks.t1 > b_start:
                continue
            if (ks.t1 - ks.t0) > INSERT_MAX_SEG_DUR_S:
                continue
            if not ks.text.strip():
                continue

            if overlaps_any_whisper(
                ks, w_windows, max_overlap=INSERT_MIN_OVERLAP_WITH_WHISPER
            ):
                continue

            inserts.append(
                SrtItem(
                    idx=0,
                    start=seconds_to_srt_time(ks.t0),
                    end=seconds_to_srt_time(ks.t1),
                    text=ks.text.strip(),
                )
            )
            used_k.add(ks.i)
            inserted_here += 1

            if out_compare_log:
                log.append(
                    f"[INSERT] {seconds_to_srt_time(ks.t0)} --> {seconds_to_srt_time(ks.t1)}"
                )
                log.append(f"  kotoba: {ks.text.strip()}")
                log.append("")

    combined = final + inserts
    combined.sort(key=lambda x: srt_time_to_seconds(x.start))

    # ---- Ollama proofreading (optional) ----
    if ollama_cfg is not None:
        print(
            f"[subloom] Ollama proofreading: model={getattr(ollama_cfg, 'model', '?')}, style={getattr(ollama_cfg, 'style', '?')}"
        )
        combined = apply_ollama_to_srt_items(combined, ollama_cfg)
    # --------------------------------------

    write_srt_items(combined, out_final_srt)

    if out_compare_log:
        out_compare_log.write_text("\n".join(log).rstrip() + "\n", encoding="utf-8")
