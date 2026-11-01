# worker-eval 20261101T084057Z

- harness: **on**, arms: routed, cases: 1 x 2 repeats
- judge: disabled (deterministic metrics only)

## Overall

| arm | success | bug_recall | bug_precision | citation_valid_rate | fallback | truncated | latency_p50_ms | latency_p95_ms | prompt_tokens | thinking_tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| routed | 1.00 | 1.00 | 0.67 | 0.72 | 1.00 | 0.00 | 7841 | 10775 | 33910 | 0 |

## By task type

### routed

| task_type | cases | key metrics |
|---|---|---|
| review | 2 | success=1.00, bug_recall=1.00, bug_precision=0.67, citation_valid_rate=0.72 |

## Cases

| arm | case | answered by | latency ms | scores | notes |
|---|---|---|---|---|---|
| routed | review-large-l3#1 | groq+gemini-lite | 11101 | fallback=1.00, truncated=0.00, bug_recall=1.00, bug_precision=0.67, citation_valid_rate=0.78 | citation_unverified; fallback_used; review_batched |
| routed | review-large-l3#2 | groq+gemini-lite | 4580 | fallback=1.00, truncated=0.00, bug_recall=1.00, bug_precision=0.67, citation_valid_rate=0.67 | citation_unverified; fallback_used; review_batched |
