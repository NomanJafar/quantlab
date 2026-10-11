**tokenbench run** · model: Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 · mode: closed · server: http://127.0.0.1:8000

| level | req | err | out tok/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | TPOT p50 |
|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 0 | 79 | 29 ms | 32 ms | 12.5 ms | 13.0 ms | 12.5 ms |
| 16 | 128 | 0 | 768 | 114 ms | 160 ms | 19.9 ms | 21.2 ms | 20.1 ms |
| 64 | 128 | 0 | 1225 | 331 ms | 520 ms | 49.0 ms | 53.3 ms | 50.0 ms |
