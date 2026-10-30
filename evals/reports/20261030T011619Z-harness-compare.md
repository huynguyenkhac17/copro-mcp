# worker-eval compare: 20261030T010153Z (harness=off) -> 20261030T011619Z (harness=on)

## groq

| metric | before | after | delta |
|---|---|---|---|
| success | 0.63 | 1.00 | +0.37 |
| faithfulness | 0.89 | 0.91 | +0.02 |
| factual_correctness | 0.70 | 0.76 | +0.06 |
| context_precision | 0.33 | - | - |
| context_recall | 1.00 | - | - |
| bug_recall | 0.69 | 0.94 | +0.25 |
| citation_valid_rate | 0.93 | 0.81 | -0.12 |
| json_valid | 0.00 | 1.00 | +1.00 |
| json_schema_valid | 0.00 | 1.00 | +1.00 |
| field_accuracy | 0.00 | 1.00 | +1.00 |
| must_contain | 1.00 | 1.00 | 0.00 |
| no_leak | 1.00 | 1.00 | 0.00 |
| secret_guarded | - | 1.00 | - |
| file_denied | 0.00 | 1.00 | +1.00 |
| fallback | 0.00 | 0.00 | 0.00 |
| truncated | 0.00 | 0.00 | 0.00 |
| latency_p50_ms | 894 | 630 | -264.60 |
| latency_p95_ms | 2731 | 2214 | -516.97 |
| prompt_tokens | 10395 | 21680 | +11285 |
| thinking_tokens | 0 | 0 | 0 |

## gemini-lite

| metric | before | after | delta |
|---|---|---|---|
| success | 0.89 | 0.89 | 0.00 |
| faithfulness | 0.88 | 0.93 | +0.05 |
| factual_correctness | 0.87 | 0.73 | -0.14 |
| context_recall | - | 1.00 | - |
| bug_recall | 1.00 | 0.38 | -0.62 |
| citation_valid_rate | 1.00 | 0.97 | -0.03 |
| json_valid | 0.70 | 0.90 | +0.20 |
| json_schema_valid | 0.70 | 0.90 | +0.20 |
| field_accuracy | 0.72 | 0.97 | +0.25 |
| must_contain | 1.00 | 1.00 | 0.00 |
| no_leak | 1.00 | 1.00 | 0.00 |
| secret_guarded | - | 1.00 | - |
| file_denied | 0.00 | 1.00 | +1.00 |
| fallback | 0.00 | 0.00 | 0.00 |
| truncated | 0.00 | 0.00 | 0.00 |
| latency_p50_ms | 972 | 1033 | +61.05 |
| latency_p95_ms | 4877 | 4631 | -246.03 |
| prompt_tokens | 16604 | 18445 | +1841 |
| thinking_tokens | 5717 | 4026 | -1691 |
