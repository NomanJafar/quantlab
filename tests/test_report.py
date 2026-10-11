
from quantlab.report import (
    build_table,
    exact_match,
    extract_gsm8k_answer,
    perplexity_from_logprobs,
    render_markdown,
)


def test_extract_gsm8k_answer_reads_final_number_after_hash_marker():
    assert extract_gsm8k_answer("Natalia sold 48/2 = 24 clips in May. 48+24 = 72. #### 72") == "72"
    assert extract_gsm8k_answer("The answer is $1,250.") == "1250"
    assert extract_gsm8k_answer("no digits here") is None


def test_exact_match_compares_normalised_numbers():
    assert exact_match("So the total is 72.", "72") is True
    assert exact_match("Answer: 72.0", "72") is True
    assert exact_match("I think 71", "72") is False


def test_perplexity_from_logprobs_is_exp_of_mean_negative_logprob():
    import math
    lps = [-1.0, -1.0, -3.0, -3.0]
    assert perplexity_from_logprobs(lps) == math.exp(2.0)
    assert perplexity_from_logprobs([]) is None


def _run(name, weights=6.0, kv=6.0, ppl=10.0, gsm=0.5, ttft1=40, itl1=15, itl16=25, tps64=900):
    return {
        "scheme": name, "model": f"Qwen/Qwen2.5-3B-{name}", "quant_arg": None if name == "fp16" else name,
        "startup": {"model_load_gib": weights, "kv_cache_gib": kv, "kv_cache_tokens": int(kv * 2**30 / 36864)},
        "quality": {"perplexity": ppl, "ppl_samples": 200, "gsm8k_exact_match": gsm, "gsm8k_n": 100},
        "latency": {"1": {"ttft_p50_ms": ttft1, "itl_p50_ms": itl1, "output_tokens_per_s": 66},
                    "16": {"ttft_p50_ms": ttft1 * 3, "itl_p50_ms": itl16, "output_tokens_per_s": 600},
                    "64": {"ttft_p50_ms": ttft1 * 6, "itl_p50_ms": itl16 * 1.5, "output_tokens_per_s": tps64}},
    }


def test_build_table_one_row_per_scheme_with_deltas_vs_fp16():
    rows = build_table([_run("fp16"), _run("awq", weights=2.1, kv=9.9, ppl=10.4, gsm=0.46, itl1=11, tps64=850)])
    assert [r["scheme"] for r in rows] == ["fp16", "awq"]
    awq = rows[1]
    assert awq["weights_gib"] == 2.1 and awq["kv_cache_gib"] == 9.9
    assert awq["ppl_delta_pct"] == 4.0 and awq["gsm8k_delta_pts"] == -4.0
    assert awq["itl1_speedup"] == round(15 / 11, 2)
    assert rows[0]["ppl_delta_pct"] == 0.0


def test_render_markdown_has_header_and_rows(tmp_path):
    md = render_markdown(build_table([_run("fp16"), _run("gptq-int8", weights=3.3)]))
    assert md.splitlines()[0].startswith("| scheme |")
    assert md.count("\n| ") == 2
    assert "GSM8K" in md and "ITL" in md


def test_build_table_tolerates_missing_quality_or_levels():
    r = _run("awq"); r["quality"] = {}; r["latency"].pop("64")
    row = build_table([_run("fp16"), r])[1]
    assert row["perplexity"] is None and row["tps64"] is None
