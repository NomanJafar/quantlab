**tokenbench run** · model: Qwen/Qwen2.5-3B-Instruct · mode: closed · server: http://127.0.0.1:8000

| level | req | err | out tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | TPOT p50 |
|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 0 | 25 | 52 ms | 54 ms | 40.3 ms | 42.3 ms | 40.1 ms |
| 16 | 128 | 0 | 411 | 152 ms | 196 ms | 37.7 ms | 39.3 ms | 37.9 ms |
| 64 | 128 | 0 | 962 | 313 ms | 501 ms | 63.5 ms | 67.8 ms | 64.3 ms |
