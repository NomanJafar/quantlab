# quantlab

Same model, FP16 vs INT8 vs AWQ (and FP8 where supported): measured quality, latency and VRAM trade-offs for LLM serving.

Project 5 of 15 in [inference-infra-portfolio](https://github.com/NomanJafar/inference-infra-portfolio).

## What this builds

Serve the same model in FP16, INT8 and AWQ (FP8 where hardware allows) and benchmark quality, latency and VRAM side by side.

## Why it matters

Quantization decisions need measured trade-offs, not vibes.

## Status

Concept primer written (`CONCEPTS.md`). The comparison notebook and report code are next; results land in `results/` once the GPU runs are done.

## Layout

```
CONCEPTS.md     the theory behind the project, written before the code
src/quantlab/   implementation
tests/          CPU-only tests, runnable on a laptop
notebooks/      GPU notebooks (Kaggle / Colab) that run the experiments and write results/
results/        measured numbers and plots
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

GPU experiments run from `notebooks/`; each notebook clones this repo, installs it and writes its outputs to `results/`.

## Hardware notes

Developed against free-tier GPUs (Kaggle and Colab T4, 16 GB). Where a T4 limits what can be measured (no FP8, Turing-era kernels), the README says so rather than papering over it.

## License

MIT
