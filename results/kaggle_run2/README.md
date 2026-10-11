Run 2 (2026-10-11): all four servers started (vLLM chose MarlinLinearKernel for AWQ and GPTQ-Int4; startup logs kept here),
then every scheme failed in the quality step with `RuntimeError: asyncio.run() cannot be called from a running event loop`:
Jupyter already owns the loop. Fix for run 3: run the eval coroutine on a worker thread with its own loop.
