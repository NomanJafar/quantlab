# Concepts: what quantization changes, and how to measure it instead of guessing

`quantlab` serves the same model in FP16, INT8 (GPTQ) and AWQ and measures three things side by side: output quality, latency under load, and VRAM. This primer is the background for reading those numbers.

## 1. Why quantize a model you are serving

Two reasons, and they are different:

1. **Memory.** Weights at 4 bits are a quarter the size of FP16. A 7B model drops from 14 GB to ~4 GB, which means it fits a smaller GPU, or, on the same GPU, the freed memory becomes KV cache (project 3) and therefore concurrency.
2. **Bandwidth.** Decode is memory-bound (servellm primer, section 1). Each decode step streams the weights through the GPU; if the weights are 4× smaller, the step is up to 4× faster *at small batch sizes*. At large batch sizes decode becomes compute-bound and the weight-size advantage fades, while the cost of de-quantizing weights before the matmul remains. So quantization helps latency most where batching helps least: interactive, low-concurrency serving.

A third reason people cite, "it's cheaper", is these two restated as $/token.

## 2. Weight-only vs weight-and-activation quantization

- **Weight-only (INT4/INT8 weights, FP16 activations):** weights are stored compressed and de-quantized to FP16 on the fly inside the matmul kernel. The arithmetic is still FP16. This is what GPTQ and AWQ do. It attacks memory and bandwidth, not compute.
- **Weight-and-activation (INT8/FP8 for both):** the matmul itself runs in low precision on tensor cores that support it (INT8 on Turing and later; FP8 on Hopper/Ada and later). This also speeds up compute-bound prefill and large-batch decode, but activations have outliers that are hard to quantize, which is why schemes like SmoothQuant exist.

On a T4 (Turing): INT8 tensor cores exist, **FP8 does not**. That is why FP8 is out of scope for this lab and the README says so instead of pretending.

## 3. The schemes compared here

**GPTQ** (Frantar et al., 2022): post-training, layer by layer. For each weight matrix it quantizes columns one at a time and uses second-order information (the Hessian of the layer's reconstruction error on a calibration set) to adjust the not-yet-quantized columns to compensate for the error introduced so far. Typically 4-bit with group size 128 (one FP16 scale per 128 weights), or 8-bit. Needs a calibration set of a few hundred sequences. Quality at 4-bit is good; at 8-bit it is near-lossless.

**AWQ** (Lin et al., 2023): observes that a small fraction of weight channels matter much more (those multiplied by large activations), and that scaling those channels up before quantization (and the corresponding activations down) protects them with no runtime cost. Also 4-bit, group-wise. Tends to match or beat GPTQ at 4-bit and its kernels are fast at small batch.

**INT8 weight-only** (via GPTQ-Int8 or vLLM's built-in `--quantization` paths): half the memory of FP16, almost no quality loss, less speedup than 4-bit.

**FP16** is the baseline: what the model was trained to produce.

The lab uses the official pre-quantized Qwen2.5 checkpoints (`-GPTQ-Int8`, `-GPTQ-Int4`, `-AWQ`) so that calibration is not a variable; quantizing weights ourselves is out of scope.

## 4. Where quality goes

Quantization error is noise added to weights. Its effect on outputs is not uniform:

- Perplexity on generic text barely moves at 8-bit and moves a little at 4-bit (typically +1–3% for a well-made 4-bit model of 7B parameters).
- **Small models lose more.** A 1.5B model has less redundancy than a 70B one; 4-bit on 1.5B is a real quality hit, which is a reason to use the 3B model for this lab.
- **Reasoning and arithmetic degrade before fluency does.** GSM8K exact-match is a sharper instrument than perplexity. A model can read fine and get more sums wrong.
- Outlier layers (often the first and last) are sometimes kept in higher precision by the quantizer for this reason.

So the lab measures both: perplexity on a fixed 200-sample text set (sensitive, cheap) and exact match on 100 GSM8K problems (coarse, but the one that matters for a reasoning workload).

## 5. Where latency goes

Expect, on a T4 with a 3B model:

- **Single stream (concurrency 1):** 4-bit noticeably faster per token than FP16, because the decode step reads a quarter of the bytes. INT8 in between.
- **At concurrency 16–64:** the gap narrows or inverts. The de-quantization work is paid per matmul regardless of batch, and once decode is compute-bound the weight size no longer sets the pace. Kernel quality matters here: a good AWQ kernel keeps its edge longer than a naive one.
- **Prefill / TTFT:** compute-bound, so weight-only quantization does not help and can hurt. If TTFT gets worse under quantization, that is the de-quantization overhead showing.

The `tokenbench` closed-loop sweep at 1, 4, 16, 64 captures all three regimes.

## 6. Where memory goes

Weights are the headline, but check the startup log: vLLM prints "Model loading took X GiB" and "Available KV cache memory: Y GiB". The second number is the one that turns into concurrency. For the 3B model in FP16, weights are ~6 GB of a 15 GB card; at 4-bit, ~2 GB. That 4 GB difference is roughly 140,000 extra KV tokens at 28 KB/token (kvcalc), which is the real production argument for quantization on small GPUs.

## 7. Making the comparison fair

- Same model family and size, same prompts, same `max_tokens`, `ignore_eos` on for the latency sweeps so output length is not a variable, `temperature 0` for the quality evals.
- Same vLLM version and flags apart from the quantization-specific ones.
- Warm-up before every measurement; each sweep level long enough for p95 to be meaningful (tokenbench primer, section 5).
- Report the kernel vLLM selected for each scheme (it logs it): comparing schemes is partly comparing kernels.
- Report what was *not* measured: FP8 (hardware), self-quantized models (calibration variance), larger models (memory).

## 8. What the experiment will show

A single table: scheme, weights GiB, KV cache GiB, perplexity, GSM8K exact match, TTFT p50 and ITL p50 at concurrency 1 and 16, output tokens/s at concurrency 64. And one plot: ITL vs concurrency for the four configurations. The write-up answers: for an interactive workload on this hardware, which scheme would you deploy, and what did it cost in accuracy?

## Further reading

- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022).
- Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (2023).
- Dettmers & Zettlemoyer, *The case for 4-bit precision: k-bit inference scaling laws* (2022): why 4-bit is usually the sweet spot.
- vLLM docs: *Quantization* section, supported kernels per GPU generation.
