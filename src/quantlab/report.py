"""Quality metrics and the comparison table for the quantization lab.

Pure functions on dicts so they are testable without a GPU. The notebook produces
one `run` dict per scheme (startup facts, quality numbers, tokenbench levels);
`build_table` turns the list into rows with deltas against the FP16 baseline.
"""

from __future__ import annotations

import math
import re

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_gsm8k_answer(text: str) -> str | None:
    """The number after '####' if present (GSM8K convention), else the last number in the text."""
    if "####" in text:
        text = text.split("####")[-1]
    nums = _NUM.findall(text)
    if not nums:
        return None
    return _normalise(nums[-1])


def _normalise(num: str) -> str:
    num = num.replace(",", "").strip()
    try:
        f = float(num)
    except ValueError:
        return num
    return str(int(f)) if f == int(f) else str(f)


def exact_match(prediction: str, gold: str) -> bool:
    p = extract_gsm8k_answer(prediction)
    return p is not None and p == _normalise(gold)


def perplexity_from_logprobs(logprobs: list[float]) -> float | None:
    if not logprobs:
        return None
    return math.exp(-sum(logprobs) / len(logprobs))


def build_table(runs: list[dict]) -> list[dict]:
    base = next((r for r in runs if r["scheme"] == "fp16"), runs[0])
    bq, bl = base.get("quality", {}), base.get("latency", {})
    rows = []
    for r in runs:
        q, lat, st = r.get("quality", {}), r.get("latency", {}), r.get("startup", {})

        def g(level, key, _lat=lat):
            return _lat.get(str(level), {}).get(key)

        ppl, b_ppl = q.get("perplexity"), bq.get("perplexity")
        gsm, b_gsm = q.get("gsm8k_exact_match"), bq.get("gsm8k_exact_match")
        itl1, b_itl1 = g(1, "itl_p50_ms"), bl.get("1", {}).get("itl_p50_ms")
        rows.append({
            "scheme": r["scheme"], "model": r.get("model"),
            "weights_gib": st.get("model_load_gib"), "kv_cache_gib": st.get("kv_cache_gib"),
            "kv_cache_tokens": st.get("kv_cache_tokens"),
            "perplexity": ppl, "ppl_delta_pct": round((ppl / b_ppl - 1) * 100, 1) if ppl and b_ppl else None,
            "gsm8k": gsm, "gsm8k_delta_pts": round((gsm - b_gsm) * 100, 1) if gsm is not None and b_gsm is not None else None,
            "ttft1_ms": g(1, "ttft_p50_ms"), "itl1_ms": itl1,
            "itl1_speedup": round(b_itl1 / itl1, 2) if itl1 and b_itl1 else None,
            "ttft16_ms": g(16, "ttft_p50_ms"), "itl16_ms": g(16, "itl_p50_ms"),
            "tps64": g(64, "output_tokens_per_s"),
        })
    return rows


def _f(x, d=1, suffix=""):
    return "–" if x is None else f"{x:.{d}f}{suffix}"


def render_markdown(rows: list[dict]) -> str:
    head = ["scheme", "weights GiB", "KV cache GiB", "perplexity (Δ%)", "GSM8K EM (Δpts)",
            "TTFT p50 @1", "ITL p50 @1 (speedup)", "ITL p50 @16", "out tok/s @64"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append("| " + " | ".join([
            r["scheme"], _f(r["weights_gib"], 2), _f(r["kv_cache_gib"], 2),
            f"{_f(r['perplexity'], 2)} ({_f(r['ppl_delta_pct'], 1, '%')})",
            f"{_f(r['gsm8k'] * 100 if r['gsm8k'] is not None else None, 0, '%')} ({_f(r['gsm8k_delta_pts'], 1)})",
            _f(r["ttft1_ms"], 0, " ms"), f"{_f(r['itl1_ms'], 1, ' ms')} ({_f(r['itl1_speedup'], 2, 'x')})",
            _f(r["itl16_ms"], 1, " ms"), _f(r["tps64"], 0),
        ]) + " |")
    return "\n".join(lines) + "\n"
