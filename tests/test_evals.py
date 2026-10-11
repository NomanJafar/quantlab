import math

from quantlab.evals import (
    GSM8K_INSTRUCTION,
    gsm8k_request_body,
    logprobs_from_ppl_response,
    ppl_request_body,
    score_gsm8k,
)


def test_ppl_request_asks_for_prompt_logprobs_and_no_generation():
    b = ppl_request_body("m", "hello world")
    assert b["prompt_logprobs"] == 0 and b["max_tokens"] == 1 and b["temperature"] == 0


def test_logprobs_from_ppl_response_skips_first_token_and_takes_token_logprob():
    resp = {"choices": [{"prompt_logprobs": [None, {"42": {"logprob": -0.5, "rank": 1}}, {"7": {"logprob": -2.0, "rank": 1}}]}]}
    assert logprobs_from_ppl_response(resp) == [-0.5, -2.0]
    assert logprobs_from_ppl_response({"choices": [{"prompt_logprobs": None}]}) == []


def test_gsm8k_request_is_greedy_and_instructs_the_marker():
    b = gsm8k_request_body("m", "What is 2+2?")
    assert b["temperature"] == 0 and b["messages"][0]["content"].startswith(GSM8K_INSTRUCTION)
    assert "####" in b["messages"][0]["content"]


def test_score_gsm8k_counts_exact_matches():
    s = score_gsm8k(["... #### 72", "the answer is 5", "#### 1,000"], ["72", "6", "1000"])
    assert s["gsm8k_exact_match"] == 2 / 3 and s["gsm8k_n"] == 3 and s["hits"] == [True, False, True]
    assert math.isclose(s["gsm8k_exact_match"], 0.6667, rel_tol=1e-3)
