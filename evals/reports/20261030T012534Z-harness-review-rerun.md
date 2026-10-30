# worker-eval 20261030T012534Z

- harness: **on**, arms: gemini-lite, groq, cases: 4
- judge: disabled (deterministic metrics only)

## Overall

| arm | success | bug_recall | citation_valid_rate | json_valid | json_schema_valid | fallback | truncated | latency_p50_ms | latency_p95_ms | prompt_tokens | thinking_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini-lite | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 3402 | 6159 | 3580 | 3223 |
| groq | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 2116 | 2660 | 4080 | 0 |

## By task type

### gemini-lite

| task_type | cases | key metrics |
|---|---|---|
| review | 4 | success=1.00, bug_recall=1.00, citation_valid_rate=1.00, json_valid=1.00, json_schema_valid=1.00 |

### groq

| task_type | cases | key metrics |
|---|---|---|
| review | 4 | success=1.00, bug_recall=1.00, citation_valid_rate=1.00, json_valid=1.00, json_schema_valid=1.00 |

## Cases

| arm | case | answered by | latency ms | scores | notes |
|---|---|---|---|---|---|
| gemini-lite | review-orders-l3 | gemini-lite | 3103 | fallback=0.00, truncated=0.00, bug_recall=1.00, citation_valid_rate=1.00 |  |
| gemini-lite | review-orders-l2 | gemini-lite | 1458 | fallback=0.00, truncated=0.00, bug_recall=1.00, citation_valid_rate=1.00 |  |
| gemini-lite | review-orders-json | gemini-lite | 6593 | fallback=0.00, truncated=0.00, json_valid=1.00, json_schema_valid=1.00, bug_recall=1.00, citation_valid_rate=1.00 |  |
| gemini-lite | review-ratelimiter-js | gemini-lite | 3700 | fallback=0.00, truncated=0.00, bug_recall=1.00, citation_valid_rate=1.00 |  |
| groq | review-orders-l3 | groq | 2562 | fallback=0.00, truncated=0.00, bug_recall=1.00, citation_valid_rate=1.00 |  |
| groq | review-orders-l2 | groq | 1062 | fallback=0.00, truncated=0.00, bug_recall=1.00, citation_valid_rate=1.00 |  |
| groq | review-orders-json | groq | 2677 | fallback=0.00, truncated=0.00, json_valid=1.00, json_schema_valid=1.00, bug_recall=1.00, citation_valid_rate=1.00 |  |
| groq | review-ratelimiter-js | groq | 1670 | fallback=0.00, truncated=0.00, bug_recall=1.00, citation_valid_rate=1.00 |  |
