# worker-eval compare: 20261101T080853Z (harness=on) -> 20261101T081155Z (harness=on)

## routed

| metric | before | after | delta |
|---|---|---|---|
| success | 1.00 | 1.00 | 0.00 |
| bug_recall | 0.72 | 1.00 | +0.28 |
| bug_precision | 1.00 | 0.88 | -0.12 |
| citation_valid_rate | 0.96 | 0.96 | +0.01 |
| fallback | 0.67 | 0.50 | -0.17 |
| truncated | 0.00 | 0.00 | 0.00 |
| latency_p50_ms | 1737 | 3217 | +1480 |
| latency_p95_ms | 2729 | 13251 | +10522 |
| prompt_tokens | 51272 | 55858 | +4586 |
| thinking_tokens | 0 | 0 | 0 |

## groq

| metric | before | after | delta |
|---|---|---|---|
| success | 0.50 | 0.50 | 0.00 |
| bug_recall | 1.00 | 1.00 | 0.00 |
| bug_precision | 1.00 | 1.00 | 0.00 |
| citation_valid_rate | 1.00 | 1.00 | 0.00 |
| fallback | 0.00 | 0.00 | 0.00 |
| truncated | 0.00 | 0.00 | 0.00 |
| latency_p50_ms | 2464 | 3738 | +1274 |
| latency_p95_ms | 2529 | 4150 | +1621 |
| prompt_tokens | 3171 | 3705 | +534 |
| thinking_tokens | 0 | 0 | 0 |
