**tokenbench run** · model: Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8 · mode: closed · server: http://127.0.0.1:8000

| level | req | err | out tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | TPOT p50 |
|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 0 | 59 | 31 ms | 34 ms | 16.8 ms | 17.3 ms | 16.7 ms |
| 16 | 128 | 0 | 636 | 122 ms | 176 ms | 24.1 ms | 25.5 ms | 24.3 ms |
| 64 | 128 | 0 | 1126 | 382 ms | 551 ms | 53.5 ms | 57.7 ms | 54.3 ms |
