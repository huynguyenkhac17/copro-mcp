# Copro

[![CI](https://github.com/huynguyenkhac17/copro-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/huynguyenkhac17/copro-mcp/actions/workflows/ci.yml)

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-1.2.0+-green.svg)](https://modelcontextprotocol.io/)

An MCP (Model Context Protocol) server acting as a middleware between Claude Code and a pool of local/cloud LLM workers.

This project is **not a chatbot**. Claude Code remains the ultimate decision-making supervisor; the local models act as specialized "workers" used to compress context (reading multiple files, summarizing, finding bug candidates, initial planning) or handle low/medium effort tasks. This significantly saves Claude's context window and API token usage.

The server exposes three MCP tools: `delegate_task` does the work, `check_job_status` collects the result of an `async_mode` call, and `get_gateway_stats` reports usage. Claude does not need to know which model runs a task or with which generation parameters: it picks a budget (`L1`-`L3`) and, optionally, a `task_type`; routing, fallback, quotas and retries happen behind that interface.

## Architecture

```
Claude Code (supervisor)
      │ delegate_task(task, budget, task_type, context, file_paths, urls, output_format, json_schema)
      ▼
MCP layer        server.py — registers the tool, maps inputs/exceptions
      ▼
Application layer  DelegateTaskUseCase — orchestrates the main flow
      ▼
Domain layer      interfaces / models / PromptBuilder / OutputParser
      ▼
Provider layer    ProviderFactory → OllamaProvider / OpenAICompatibleProvider
```

**Core Principles:**
- The domain layer has zero external dependencies.
- The application layer only interacts with the domain via the `LLMProvider` interface.
- The MCP layer handles wiring and input validation with no business logic.
- Adding a new provider or budget requires **no existing code changes** (see [Extensibility](#extensibility)).

## Directory Structure

```
mcp/
├── pyproject.toml
├── .env.example
├── src/copro/
│   ├── server.py           # MCP entrypoint, registers delegate_task tool
│   ├── application/        # Orchestration layer (DTOs, Use Cases, Map-Reduce, Jobs)
│   ├── domain/             # Core business logic (Interfaces, Prompts, Parsers, RAG)
│   ├── providers/          # LLM integrations (Ollama, OpenAI-compatible, Cache)
│   ├── config/             # Pydantic Settings + the shipped YAML defaults:
│   │                       #   default.yaml, budgets.yaml, task_profiles.yaml
│   ├── evals/              # copro eval
│   └── infra/              # Logging, Retries, Cache, Vector Store, Stats
├── evals/                  # eval datasets and reports
└── tests/
    ├── unit/               # Uses mocked LLMProvider
    └── integration/        # Uses real Ollama (auto-skips if unavailable)
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (or Python >= 3.11 with pip).
- At least one cloud API key. With the shipped config, L1-L3 requests go to Groq
  first and fall back through the other vendors:

  | Provider | Key variable | Cost | Notes |
  |---|---|---|---|
  | `groq` | `GROQ_API_KEY` | free tier | [console.groq.com](https://console.groq.com/keys); 8K tokens/minute per organization, prompts over ~6K tokens skip it |
  | `gemini-lite`, `gemini` | `GEMINI_API_KEY` | free tier | [Google AI Studio](https://aistudio.google.com/apikey); quota is per Google Cloud project |
  | `mistral` *(experimental)* | `MISTRAL_API_KEY` | free tier | [console.mistral.ai](https://console.mistral.ai/); low request rate on the free plan |
  | `deepseek` *(experimental)* | `DEEPSEEK_API_KEY` | paid, cheap | [platform.deepseek.com](https://platform.deepseek.com/); 1M context, strong on code; placed after the free tiers |
  | `openrouter` *(experimental)* | `OPENROUTER_API_KEY` | free models | [openrouter.ai](https://openrouter.ai/keys); 50 requests/day (1000 after buying 10 credits) |

  *Experimental* providers are configured from their vendors' documentation and covered
  by unit tests of the request they send, but have not yet been run against the live
  API or measured with `copro eval`. Reports of what works and what does not are welcome.

  A provider whose key is missing is skipped with a warning and dropped from the
  chains; the others keep working. A call routed to it explicitly fails with
  `Provider '<name>' is configured but was skipped at startup: ...`, naming the
  environment variables it looks for. More keys for the same vendor go in `_2`,
  `_3`... variables (see `api_key_envs` in `default.yaml`).
- *(Optional)* [Ollama](https://ollama.com) for `provider="ollama"` (content that must
  not leave the machine) and for RAG embeddings (`ollama pull nomic-embed-text`).

## Installation

Install the server as a uv tool, straight from the repository:

```bash
uv tool install git+https://github.com/huynguyenkhac17/copro-mcp
```

This puts two commands on your PATH: `copro-mcp`, the MCP server, and `copro`, with
`copro serve` (the same server), `copro stats`, `copro metrics` and `copro eval`.
`uv tool upgrade copro-mcp` updates it later. Without installing anything,
`uvx --from git+https://github.com/huynguyenkhac17/copro-mcp copro-mcp` runs it from
uv's cache.

Then register it with Claude Code, passing the keys and the directories the server may
read (`file_paths` is disabled until you name at least one):

```bash
claude mcp add -s user copro \
  -e GROQ_API_KEY=gsk_... \
  -e GEMINI_API_KEY=AIza... \
  -e 'COPRO_FILE_ACCESS__ALLOWED_ROOTS=["/home/me/projects"]' \
  -- copro-mcp
```

Instead of `-e`, the same variables can go in `~/.copro/.env`, which the
server reads at startup (see `.env.example`).

**From a source checkout** (development):

```bash
git clone https://github.com/huynguyenkhac17/copro-mcp && cd copro-mcp
uv venv && uv pip install -e ".[dev]"
cp .env.example .env    # keys and allowed_roots for this checkout
claude mcp add -s user copro -- "$PWD/.venv/bin/copro-mcp"
```

On Windows the script is `.venv\Scripts\copro-mcp.exe`. Code changes in `src/`
take effect when the server restarts (`/mcp` in Claude Code reconnects it).

### Upgrading from llm-worker-pool (before 0.4.0)

The project was called `llm-worker-pool`. Until 0.4.x ends the old names still work,
each with a note on stderr:

- `LLM_WORKER_POOL_*` variables are read as `COPRO_*` when the new name is not set.
- `~/.llm_worker_pool/` is copied to `~/.copro/` on the first start (the old one is left
  in place and no longer read).
- `llm-worker-pool`, `worker-stats`, `worker-metrics` and `worker-eval` run
  `copro-mcp`, `copro stats`, `copro metrics` and `copro eval`.

Re-register the server under its new name (`claude mcp remove llm-worker-pool`, then
the `claude mcp add` above); its tools become `mcp__copro__*`.

## Configuration

Everything is configuration-driven. **No code modifications are required.**

- `src/copro/config/default.yaml`: providers, timeouts, retries, concurrency,
  logging, file access, caching, background jobs and RAG.
- `src/copro/config/budgets.yaml`: execution budgets. Add a budget by adding
  an entry.
- `src/copro/config/task_profiles.yaml`: kinds of work (`task_type`).

These ship inside the package. To change them for an installed tool, copy one and
point the server at the copy with `COPRO_CONFIG`,
`COPRO_BUDGETS_CONFIG` or `COPRO_TASK_PROFILES_CONFIG`. Single
settings are overridden with environment variables (prefix `COPRO_`, `__` for
nested fields, e.g. `COPRO_FILE_ACCESS__ALLOWED_ROOTS`). Priority (highest
to lowest):
`init kwargs > real environment variables > .env files > default.yaml`

The `.env` files are the checkout's own `<repo>/.env` (when running from a source
checkout) and then `~/.copro/.env`.

> Note: provider API key variables (`api_key_env`/`api_key_envs`, e.g. `GEMINI_API_KEY`) follow the same priority but are a special case: their variable *names* are arbitrary (whatever you write in `default.yaml`), not `COPRO_`-prefixed settings. The `.env` files are loaded via `python-dotenv` directly into the process environment at startup, so setting them there works exactly like exporting real OS environment variables. A real OS env var of the same name still wins if both are set.

### Providers

`default.yaml` declares multiple providers simultaneously. Each provider has a name and a `type` (the implementation class in `providers/registry.py`):

```yaml
default_provider: gemini-lite
providers:
  ollama:
    type: ollama
    endpoint: http://localhost:11434
    model: qwen2.5-coder:7b
  gemini:
    type: openai_compatible   # Compatible with Gemini/OpenAI/Groq
    endpoint: https://generativelanguage.googleapis.com/v1beta/openai
    model: gemini-flash-latest
    api_key_env: GEMINI_API_KEY   # The ENV VAR name containing the key
    # api_key_envs: [GEMINI_API_KEY_1, GEMINI_API_KEY_2]  # optional: round-robin across multiple keys
    # concurrency: 10   # optional: max parallel requests for this provider
```

The `delegate_task` tool accepts a `provider: str | None` parameter to select the backend per invocation (defaults to `default_provider`). Routing is decided by **confidentiality, not by whether data is on local disk**: send only genuinely sensitive content (secrets, credentials, proprietary data that must not leave the machine) to `ollama`, and route **everything else** — including summarizing ordinary local project code — to a cloud provider (the default). Note that `file_paths` reads files server-side and works with *any* provider, so summarizing a local codebase on the cloud is fully supported; cloud is in fact preferred for multi-file work because it runs many workers in parallel (concurrency + multi-key round-robin), whereas `ollama` is serial by default. For large local codebases, combine `file_paths` with `map_reduce=true` on a cloud provider.

Each provider runs behind its own concurrency limit (`ConcurrencyProvider`), so a slow/serial local model never blocks parallel calls to a cloud provider (e.g. during Map-Reduce). If `concurrency` is omitted, it defaults by provider `type`: `ollama` -> 1 (VRAM-safe), everything else -> 5. When `api_key_envs` lists multiple keys, requests round-robin across them to spread load/rate limits.

### Budget Profiles

Tuned for the cloud default provider. The earlier values were sized for a local 9B on
limited VRAM, where a small `num_predict` was the cheapest guard against runaway
generation; on a cloud model that same ceiling became the main cause of truncated answers.

| Budget | Purpose | reasoning_effort | num_ctx | num_predict | timeout |
|--------|---------|------------------|---------|-------------|---------|
| `L1` | Fast, deterministic — categorization, extraction, short answers | low | 131072 | 4096 | 45s |
| `L2` | Balanced — summarizing, moderate refactoring, code explanation | low | 262144 | 16384 | 120s |
| `L3` | Highest quality — architecture analysis, bug finding, reading multiple files | medium | 524288 | 32768 | 300s |

**Thinking shares the output budget.** Reasoning models bill internal thought against
`num_predict`, so a low ceiling starves the visible answer. Measured on
`gemini-flash-latest` at `max_tokens=1024` with no reasoning cap: 979 tokens went to
thinking, 41 to the answer, and it still stopped at `finish_reason=length`.

Two fields control this and each backend reads only its own, declared via the
provider's `thinking_option`:

| Field | Backend | Values |
|-------|---------|--------|
| `think` | Ollama | `true` / `false` |
| `reasoning_effort` | OpenAI-compatible (Gemini, ...) | `low` / `medium` / `high` — `none` is rejected with HTTP 400, so `low` is the floor |

Setting `think: false` does nothing on a cloud provider: that field does not exist in
the OpenAI-compatible wire format and is never sent.

Every response reports `prompt_tokens`, `completion_tokens`, `thinking_tokens`
(derived as total − prompt − completion, because providers bill thinking inside the
total while reporting it in neither part) and `truncated`. An answer cut off at the
ceiling carries an explicit `output_truncated_at_max_tokens` warning instead of
passing as a clean success.

### Direct File Access (`file_paths`)

Instead of Claude reading files and pasting them into the prompt (wasting tokens), the server can read files directly via absolute paths. This is **disabled by default** for safety: the shipped `allowed_roots` is empty. Name the directories per machine, as a JSON array:

```bash
COPRO_FILE_ACCESS__ALLOWED_ROOTS=["/home/me/projects", "/srv/repo"]
```

`max_file_bytes` (200 KB per file) and `max_total_bytes` (800 KB per call) cap what is read. Keep the roots narrow: the deny-list below blocks `.env*`, keys and `.git`, but not every credential file a home directory can hold (`~/.aws/credentials`, `~/.netrc`, `~/.git-credentials`...).

An item may be a file, a directory (walked recursively) or a glob such as `D:/repo/src/**/*.py`. While walking, `.git`, `node_modules`, `.venv`, build output and lockfiles are skipped, and symlinked or junction directories are never entered. Files are taken most useful first (README, source, tests, docs, then the rest), so when the byte budget runs out the noise is what gets dropped. `max_dir_files`, `max_dir_depth` and `max_dir_entries` cap each item.

Errors (path traversal, file not found, too large, unreadable) are gracefully handled and appended to `DelegateTaskResult.warnings`. Files skipped inside a directory or glob are reported as one count per item. When `file_paths` is the request's only material and none of it can be read, the call fails before any model is asked (`success: false`, error `None of file_paths could be read: ...`, the same warnings attached) instead of returning the model's answer to the bare task.

### Follow-ups (`continuation_id`)

Every answer carries a `continuation_id`. Pass it back with a new `task` to follow up without resending the context. This is useful after `output_truncated_at_max_tokens` (`task="continue"`), or to ask a second question about the same files. The server replays the conversation's context and earlier turns as a transcript, and older turns are dropped first when they do not fit `num_ctx`. Provider, budget, task_type, output_format and json_schema are inherited unless you set them to another value (leaving one out, or passing null, keeps the inherited value; start a new conversation to drop a task_type). New `context`/`file_paths`/`urls` are added to the conversation.

Conversations live in server memory (`conversations:` in `default.yaml`: 1 h TTL, 50 conversations, 20 turns). After expiry or a restart the call fails with `continuation_not_found`. A follow-up always runs on one provider, and it cannot be combined with `map_reduce`.

### Guardrails

`allowed_roots` can be a whole drive, so two more checks sit on top of it (`guardrails` in `default.yaml`):

- **Deny-list.** Files matching `.env*`, `*.pem`, `*.key`, `*id_rsa*`, `*id_ed25519*`, `credentials*.json` or `**/.git/**` are never read, even when a directory is listed or a symlink points at them. The warning is `file_denied_sensitive:<path>`. Use `extra_deny_globs` to add patterns, or `deny_globs` to replace the list.
- **Secret redaction.** If **any** tier in the fallback chain is a cloud provider, secret-looking literals in the task and context are replaced with `[REDACTED:<kind>]` before the call (`secret_redacted:<kind> xN`). This covers `AIza…`, `gsk_…`, `sk-…`, `ghp_…`, PEM private keys, `password="…"`, `SECRET_TOKEN=…` and `user:pass@` in URLs. Nothing is redacted when the whole chain is local (ollama). Answers are scanned too, and a secret found there is flagged as `secret_in_response:<kind> xN`.
- **Untrusted context.** File, URL and context text is wrapped in `<untrusted_context>…</untrusted_context>`, and a grounding rule tells the worker to treat it as data, not instructions. This guards against prompt injection hidden in a README or web page.

`DelegateTaskResult.guardrail_actions` lists what was done on each call. These checks match patterns, so they catch the common cases but are not a guarantee. For genuinely confidential data, use `provider='ollama'`.

### Task Types (`task_type`)

`task_profiles.yaml` defines kinds of work: `review`, `summarize`, `extract`, `classify`, `explain`, `translate` and `generate`. Each profile can set:

- `system_addon`: an output contract added to the system prompt.
- `default_budget`, `output_format` and `json_schema`: used only when the caller does not set them.
- `validators`:
  - `json_repair`: if the JSON fails to parse or fails the schema, the same tier gets one more try with its own output and the errors (`json_repaired` on success).
  - `citations`: every `path:line` or `{file, line, evidence_quote}` in the answer is checked against the files that were in the context. The results go to `result.verification` (`ok` / `file_not_in_context` / `line_out_of_range` / `quote_not_found`), and each failed one gets a `citation_unverified:<file>:<line>` warning.

For example, `review` returns `findings: [{file, line, severity, claim, evidence_quote}]` at L3. Leaving out `task_type` gives exactly the old behaviour. To add a type, add a YAML entry; no code change is needed.

### Local RAG

RAG is off by default. It needs the `[rag]` extra (faiss, numpy: `uv tool install "copro-mcp[rag] @ git+..."` or `pip install -e ".[rag]"`) and a local Ollama; turn it on with `COPRO_RAG__ENABLED=true`. Without the extra the server starts with RAG off and logs why.

If `file_paths` includes a directory or glob, `rag.enabled` is `true`, and the files' full text would not fit the budget's context window (`num_ctx` minus the 25% reserve), the server embeds them with `rag.embedding_model` via the Ollama at `rag.embedding_endpoint`. A local FAISS store then picks the `top_k` chunks most relevant to the `task`. Below that size the model gets every file verbatim, which is what review needs. If Ollama is unreachable, the call reports `rag_indexing_failed` and falls back to the full text.

### Response Caching

Responses are cached based on a hash of the final built prompt, generation options, and model. This avoids repeated inference costs for identical queries. The cache is in-memory and configurable in `default.yaml`:

```yaml
cache:
  enabled: true
  max_entries: 256
  ttl_seconds: 3600
```

### Background Tasks (`async_mode` & `check_job_status`)

Large tasks (Budget L3 or Map-Reduce) may exceed Claude's timeout limit. Calling `delegate_task(..., async_mode=true)` immediately returns a `job_id`. Claude can then periodically poll the `check_job_status(job_id)` tool to retrieve the results once completed.

### Map-Reduce for Large Contexts

A tier whose `max_prompt_tokens` the prompt exceeds is dropped from the fallback chain before any call (warning `prompt_cap_skip:<provider> (~N tokens > max_prompt_tokens=M)`), instead of failing locally and showing up as `fallback_used` plus a logged failure. The next tier keeps the reasoning effort of its configured position (a second tier is still curbed by `fallback_reasoning_effort`). If every tier is over its cap, nothing is dropped and the refusal reaches the caller.

A review with the findings contract (`task_type="review"`) that runs on **one** provider (`cross_check=false`, only one cross-check provider up, or a provider named in the request) is batched instead: when its files exceed that provider's `max_prompt_tokens` (groq: 6000), they go out in batches of whole files that fit, each batch with its own fallback chain, and the findings are concatenated and re-verified against all the files (warning `review_batched:<n>`, failed batches `review_batch_failed:<i>/<n>:...`). There is no LLM reduce step, so file/line/quote evidence survives. A single file over the cap is not split; its batch goes to the next tier like any oversized prompt. On `review-large-l3` this took routed recall from 0.33-0.50 to 1.00 (3 runs each), with precision 0.60-1.00 and more groq rate limiting.

`delegate_task(..., map_reduce=true)` automatically splits massive contexts (or large sets of files) into chunks that fit the budget's `num_ctx`. It processes each chunk concurrently (Map phase, bounded by the target provider's `concurrency`), then synthesizes the results (Reduce phase). This requires multiple LLM calls and is **opt-in only**.

## Registering with Claude Code

See [Installation](#installation). After changing code or config, `/mcp` in Claude Code
reconnects the server, which reloads both.

## Running Tests

```bash
# In the source checkout's venv (installed with ".[dev]", see Installation)
# Unit tests (mocks LLM, does not require Ollama)
python -m pytest tests/unit

# Integration tests (requires Ollama running - auto-skips if unavailable)
python -m pytest tests/integration
```

## Usage Stats

The server keeps counters in memory and exposes them through the `get_gateway_stats`
MCP tool — but reading them that way costs a model round-trip just to look at numbers.
Every update is therefore also mirrored, atomically, to `~/.copro/stats.json`,
which anything can read directly:

```powershell
copro stats            # human-readable table
copro stats --json     # raw snapshot, for piping into jq
```

Tracked: requests, failures, cache hits, **tokens actually spent** split into
prompt / completion / thinking, and the count of answers cut off at the output ceiling.
Cache hits are excluded from spend and counted separately as tokens saved.

Counters reset when the server restarts, since they live in the server process.

### Key state shared between sessions

Every Claude Code session starts its own server process. What one of them learns about a
key — blocked by a 429 (Retry-After / retryDelay / per-day quota), a token window drained
per Groq's `x-ratelimit-*` headers, a model that keeps hanging, a breaker quarantine —
goes to `~/.copro/key_state.json`, and the others skip that key or tier too
instead of collecting their own 429. A skipped tier says why in `fallback_used`:
`quota_exhausted_locally` (no key has quota left), `keys_quarantined_locally` (every key
quarantined by the breaker after repeated failures — connection errors, 5xx) or
`model_unresponsive_locally` (repeated timeouts). Keys are stored as a truncated SHA-256, times as
wall-clock. In-flight reservations and the counters above stay per process;
`continuation_id` conversations are never written to disk. Turn it off with
`quota.shared_state: false`.

### Progress

Long calls report progress: over MCP `notifications/progress` for a synchronous call (the
tier being tried, `map i/N done`, each cross-check arm's calls, `review batch i/N done`),
and in the `progress` field `check_job_status` returns while an async job runs.

### Retries

A 429 that says how long the key is locked is retried after exactly that long (or at once
on another free key), and given up immediately when the wait would outlast the tier's
deadline, so the fallback chain moves on.

### Prometheus metrics (`copro metrics`)

Each server instance appends one JSONL line per call (and one per failed tier) to `~/.copro/events.jsonl`. Every line is a single `O_APPEND` write, so several instances (Claude Code, Gemini CLI…) can share the file safely. It rotates to `events.jsonl.1` at `metrics.max_bytes`. Task and response text are never logged unless `logging.debug_log_content` is on. `stats.json` and `/worker` work the same as before.

A separate exporter process turns the log into metrics. It reads only the new bytes on each scrape:

```bash
uv pip install -e ".[metrics]"
copro metrics                      # http://127.0.0.1:9464/metrics
curl -s 127.0.0.1:9464/metrics | grep copro_
```

It is a separate process because MCP runs over stdio and often as several instances at once. An HTTP endpoint inside each instance would fight over the port, and each would only see its own part of the traffic.

| metric | labels |
|---|---|
| `copro_requests_total` | provider, model, budget, task_type, outcome |
| `copro_latency_seconds` (histogram) | provider, budget, outcome |
| `copro_tokens_total` | provider, kind (prompt/completion/thinking) |
| `copro_fallback_total` | requested, answered |
| `copro_errors_total` | provider, code (timeout, http_503, prompt_too_large…) |
| `copro_warnings_total` | code (fallback_used, output_truncated_at_max_tokens, json_parse_failed, secret_redacted, citation_unverified…) |
| `copro_guardrail_actions_total` | action |
| `copro_eval_score` (gauge) | metric, provider, budget, task_type (from `eval_latest.json`) |

`InvocationLogger(sinks=[...])` is a hook for a future tracer such as a self-hosted Langfuse. Nothing is installed for it yet.

## Evaluation (`copro eval`)

ragas pulls in `scikit-network`, which has no Python 3.14 wheel yet. So the eval gets its own venv on Python 3.12, and the server's `.venv` stays as it is. Without ragas, `--no-judge` still runs the deterministic metrics.

```bash
uv venv .venv-eval --python 3.12
uv pip install --python .venv-eval\Scripts\python.exe -e ".[eval,dev]"
.venv-eval\Scripts\copro eval --arm groq --arm gemini-lite --harness on   # judged run
copro eval --arm groq --arm gemini-lite --arm gemini --label baseline        # harness off
copro eval --arm groq --arm gemini-lite --harness on --compare evals/reports/<baseline>.json
copro eval --compare A.json B.json                                          # diff only
copro eval --arm groq --arm gemini-lite --harness on --emit-routing         # measured routing
copro eval --arm groq --arm gemini-lite --harness on --repeat 3 --emit-routing   # with run-to-run spread
copro eval --from-report NEW.json --gate BASELINE.json                      # regression gate
```

- **Dataset.** Cases live in `evals/datasets/*.jsonl` (fixtures in `evals/datasets/fixtures/`). Fields: `id, task, task_type, budget, file_paths|context, output_format, json_schema, reference, expected_findings, expected_fields, must_contain, must_not_contain, expect_file_denied`.
- **Arms.** Each `--arm` forces one provider with fallbacks off, so the scores belong to that model. `--arm routed` uses the real routing instead. With `--harness on`, each case's `task_type` is passed to the worker.
- **LLM-judged metrics (RAGAS).**
  - `faithfulness`: not computed for review/generate, where the claims do not come from the context.
  - `factual_correctness`: needs a `reference`.
  - `context_precision` / `context_recall`: only for cases whose `file_paths` is a directory.
  - `answer_relevancy`: needs Ollama `nomic-embed-text` at `eval.embedding_endpoint`, and is skipped when that is unreachable.
- **Judge.** The judge is `eval.judge_provider` (gemini-lite by default, since gemini-flash was too overloaded to judge on 2026-10-30; override with `--judge-provider`). It is called through the openai client with key rotation, a circuit breaker and a bounded timeout. If the judge's model is also one of the arms, the report flags the self-preference risk.
- **Deterministic metrics.** `bug_recall` (cited file plus a line within ±3), `citation_valid_rate`, `json_valid`/`json_schema_valid`, `field_accuracy`, `must_contain`, `no_leak`, `file_denied`, fallback/truncation rates, latency p50/p95 and tokens.
- **Failed calls** (429, timeout) count only towards `success`; quality metrics are averaged over answered cases, so a quota storm does not read as a quality drop.
- **Routing from eval.** `--emit-routing` writes `src/copro/config/routing_from_eval.yaml`. For each task type it picks the provider with the best primary metric (review: `bug_recall`, extract/classify: `field_accuracy`, summarize/explain: `faithfulness` or `must_contain`, translate: `factual_correctness`). Only providers that answered at least `--min-success` (0.8) of their cases qualify, and they are compared on the cases all of them answered (so a provider cannot win by failing the hard case), then on success, then on latency. Each pick carries its evidence as a comment. A task type moves away from the provider callers get today (hand-written `preferred_provider`, else the budget's provider) only when the winner leads it by more than `--switch-margin` (0.05) and by more than either provider's run-to-run spread; otherwise the line stays commented out. `--repeat N` runs every case N times per arm: a repeated case counts once at the mean of its answered runs, and the spread is how far each provider's mean moved between runs. The file fills `preferred_provider` in the task profiles, and a value written in `task_profiles.yaml` wins. Review the file before committing it.
- **Regression gate.** `--gate BASELINE.json` exits 1 when an arm's primary metric for a task type drops by more than `--gate-tolerance` (0.05). A drop in `success` is only printed as a note. `--from-report` applies `--emit-routing`/`--gate` to an existing report without calling any model.
- **Output.** Reports go to `evals/reports/<timestamp>[-label].{json,md}`. Scores are also merged into `~/.copro/eval_latest.json` for the exporter, so a `--case`/`--task-type` subset run only replaces what it measured.

### `/worker` slash command in Claude Code

`~/.claude/commands/worker.md` runs the CLI through bash injection, so the table is
produced by the command itself rather than generated by the model:

```markdown
---
description: Show Copro usage
allowed-tools: Bash(copro stats:*)
---
!`copro stats`
```

Typing `/worker` prints the current stats. The model only adds a comment when a number
warrants it — truncated answers, thinking above 40% of spend, or a provider erroring out.

## Extensibility

To **add a new provider type** (e.g., vLLM, llama.cpp):
*(Note: OpenAI, Gemini, Groq are already supported via the `openai_compatible` type)*
1. Create a class implementing the `LLMProvider` interface in `providers/`.
2. Create a builder function and register it in `providers/registry.py`.
3. Add the corresponding block to `default.yaml`.

To **add a new budget profile**: simply add an entry to `budgets.yaml`.
No code changes are required for either!
