**tokenbench run** · model: Qwen/Qwen2.5-3B-Instruct-AWQ · mode: closed · server: http://127.0.0.1:8000

| level | req | err | out tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | TPOT p50 |
|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 0 | 78 | 30 ms | 32 ms | 12.8 ms | 13.3 ms | 12.7 ms |
| 16 | 128 | 0 | 764 | 120 ms | 164 ms | 20.0 ms | 21.3 ms | 20.1 ms |
| 64 | 128 | 0 | 1218 | 380 ms | 526 ms | 49.5 ms | 53.2 ms | 50.1 ms |
