"""Quality evals over an OpenAI-compatible endpoint: perplexity via prompt logprobs,
GSM8K exact match via greedy generation. Request/response shaping is pure and tested;
the two `measure_*` coroutines just drive httpx."""

from __future__ import annotations

import asyncio

import httpx

from quantlab.report import exact_match, perplexity_from_logprobs

GSM8K_INSTRUCTION = ("Solve the problem step by step. Finish with a line of the form "
                     "'#### <number>' containing only the final numeric answer.\n\nProblem: ")


def ppl_request_body(model: str, text: str) -> dict:
    # vLLM returns per-token logprobs of the *prompt* when prompt_logprobs is set;
    # max_tokens=1 keeps generation negligible. echo is not needed with prompt_logprobs.
    return {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}


def logprobs_from_ppl_response(resp: dict) -> list[float]:
    """Flatten vLLM's prompt_logprobs (list of {token_id: {logprob,...}} or None for the first token)."""
    out = []
    for entry in resp["choices"][0].get("prompt_logprobs") or []:
        if not entry:
            continue  # first token has no conditional probability
        out.append(max(v["logprob"] for v in entry.values()))  # the actual token's entry is the max-likelihood one in rank-0 mode
    return out


def gsm8k_request_body(model: str, question: str, max_tokens: int = 400) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": GSM8K_INSTRUCTION + question}],
            "max_tokens": max_tokens, "temperature": 0, "seed": 0}


def score_gsm8k(predictions: list[str], golds: list[str]) -> dict:
    hits = [exact_match(p, g) for p, g in zip(predictions, golds, strict=True)]
    return {"gsm8k_exact_match": sum(hits) / len(hits) if hits else None, "gsm8k_n": len(hits), "hits": hits}


async def measure_perplexity(client: httpx.AsyncClient, model: str, texts: list[str], concurrency: int = 8) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def one(t):
        async with sem:
            r = await client.post("/v1/completions", json=ppl_request_body(model, t), timeout=None)
            r.raise_for_status()
            return logprobs_from_ppl_response(r.json())

    per_text = await asyncio.gather(*[one(t) for t in texts])
    all_lps = [lp for lps in per_text for lp in lps]
    return {"perplexity": perplexity_from_logprobs(all_lps), "ppl_samples": len(texts), "ppl_tokens": len(all_lps)}


async def measure_gsm8k(client: httpx.AsyncClient, model: str, items: list[dict], concurrency: int = 16) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def one(q):
        async with sem:
            r = await client.post("/v1/chat/completions", json=gsm8k_request_body(model, q), timeout=None)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    preds = await asyncio.gather(*[one(it["question"]) for it in items])
    golds = [it["answer"].split("####")[-1].strip() for it in items]
    out = score_gsm8k(list(preds), golds)
    out["examples"] = [{"q": it["question"][:120], "pred": p[-160:], "gold": g, "hit": h}
                       for it, p, g, h in list(zip(items, preds, golds, out["hits"], strict=True))[:5]]
    return out
