# LLM Worker Pool

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-1.2.0+-green.svg)](https://modelcontextprotocol.io/)

An MCP (Model Context Protocol) server acting as a middleware between Claude Code and a pool of local/cloud LLM workers.

This project is **not a chatbot**. Claude Code remains the ultimate decision-making supervisor; the local models act as specialized "workers" used to compress context (reading multiple files, summarizing, finding bug candidates, initial planning) or handle low/medium effort tasks. This significantly saves Claude's context window and API token usage.

The entire system exposes exactly **one MCP tool**: `delegate_task`. Claude does not need to know which model is running in the background or what specific text generation parameters are used—everything is abstracted behind a clean interface.

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
├── config/
│   ├── default.yaml        # providers, endpoints, timeouts, retries, logging
│   └── budgets.yaml        # L1/L2/L3 execution budget definitions
├── src/llm_worker_pool/
│   ├── server.py           # MCP entrypoint, registers delegate_task tool
│   ├── application/        # Orchestration layer (DTOs, Use Cases, Map-Reduce, Jobs)
│   ├── domain/             # Core business logic (Interfaces, Prompts, Parsers, RAG)
│   ├── providers/          # LLM integrations (Ollama, OpenAI-compatible, Cache)
│   ├── config/             # Pydantic Settings
│   └── infra/              # Logging, Retries, Cache, Vector Store, Stats
└── tests/
    ├── unit/               # Uses mocked LLMProvider
    └── integration/        # Uses real Ollama (auto-skips if unavailable)
```

## Prerequisites

- Python ≥ 3.11
- [Ollama](https://ollama.com) running locally (`ollama serve`, default `http://localhost:11434`) with a pulled model (default `qwen2.5-coder:7b`):
  ```bash
  ollama pull qwen2.5-coder:7b
  ```
- To use the `gemini` provider (the **default** provider — cloud reasoning is prioritized over the local model, which is only used for tasks that specifically need to stay local), set the `GEMINI_API_KEY` environment variable. If omitted, the gemini provider is **skipped with a warning**, and the server falls back to the local model.
- *(Optional)* To use RAG (`rag.enabled: true`), pull an embedding model (default `nomic-embed-text`):
  ```bash
  ollama pull nomic-embed-text
  ```

## Installation

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Configuration

Everything is configuration-driven. **No code modifications are required.**

- `config/default.yaml` — Providers, timeouts, retries, concurrency, logging, file access, caching, background jobs, and RAG.
- `config/budgets.yaml` — Execution budget definitions. Add a new budget by simply adding a new entry here.

You can override settings via environment variables (prefix `LLM_WORKER_POOL_`, use `__` for nested fields) or via a `.env` file at the repository root. Priority (highest to lowest):
`init kwargs > real environment variables > .env file > default.yaml`

> Note: provider API key variables (`api_key_env`/`api_key_envs`, e.g. `GEMINI_API_KEY`) follow the same priority but are a special case: their variable *names* are arbitrary (whatever you write in `default.yaml`), not `LLM_WORKER_POOL_`-prefixed settings. `.env` is loaded via `python-dotenv` directly into the process environment at startup, so setting them there works exactly like exporting real OS environment variables — a real OS env var of the same name still wins if both are set.

### Providers

`config/default.yaml` declares multiple providers simultaneously. Each provider has a name and a `type` (the implementation class in `providers/registry.py`):

```yaml
default_provider: gemini
providers:
  ollama:
    type: ollama
    endpoint: http://localhost:11434
    model: qwen2.5-coder:7b
  gemini:
    type: openai_compatible   # Compatible with Gemini/OpenAI/Groq
    endpoint: https://generativelanguage.googleapis.com/v1beta/openai
    model: gemini-2.5-flash
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

Instead of Claude reading files and pasting them into the prompt (wasting tokens), the server can read files directly via absolute paths. This is **disabled by default** for safety. Configure the allowed directories in `config/default.yaml`:

```yaml
file_access:
  allowed_roots: ["D:/project/mcp"]   # Empty = disabled
  max_file_bytes: 200000              # Limit per file
  max_total_bytes: 800000             # Limit per invocation
```

An item may be a file, a directory (walked recursively) or a glob such as `D:/repo/src/**/*.py`. While walking, `.git`, `node_modules`, `.venv`, build output and lockfiles are skipped, and symlinked or junction directories are never entered. Files are taken most useful first (README, source, tests, docs, then the rest), so when the byte budget runs out the noise is what gets dropped. `max_dir_files`, `max_dir_depth` and `max_dir_entries` cap each item.

Errors (path traversal, file not found, too large, unreadable) are gracefully handled and appended to `DelegateTaskResult.warnings`. Files skipped inside a directory or glob are reported as one count per item.

### Follow-ups (`continuation_id`)

Every answer carries a `continuation_id`. Pass it back with a new `task` to follow up without resending the context. This is useful after `output_truncated_at_max_tokens` (`task="continue"`), or to ask a second question about the same files. The server replays the conversation's context and earlier turns as a transcript, and older turns are dropped first when they do not fit `num_ctx`. Provider, budget, task_type, output_format and json_schema are inherited unless you set them to another value (leaving one out, or passing null, keeps the inherited value; start a new conversation to drop a task_type). New `context`/`file_paths`/`urls` are added to the conversation.

Conversations live in server memory (`conversations:` in `config/default.yaml`: 1 h TTL, 50 conversations, 20 turns). After expiry or a restart the call fails with `continuation_not_found`. A follow-up always runs on one provider, and it cannot be combined with `map_reduce`.

### Guardrails

`allowed_roots` can be a whole drive, so two more checks sit on top of it (`guardrails` in `config/default.yaml`):

- **Deny-list.** Files matching `.env*`, `*.pem`, `*.key`, `*id_rsa*`, `*id_ed25519*`, `credentials*.json` or `**/.git/**` are never read, even when a directory is listed or a symlink points at them. The warning is `file_denied_sensitive:<path>`. Use `extra_deny_globs` to add patterns, or `deny_globs` to replace the list.
- **Secret redaction.** If **any** tier in the fallback chain is a cloud provider, secret-looking literals in the task and context are replaced with `[REDACTED:<kind>]` before the call (`secret_redacted:<kind> xN`). This covers `AIza…`, `gsk_…`, `sk-…`, `ghp_…`, PEM private keys, `password="…"`, `SECRET_TOKEN=…` and `user:pass@` in URLs. Nothing is redacted when the whole chain is local (ollama). Answers are scanned too, and a secret found there is flagged as `secret_in_response:<kind> xN`.
- **Untrusted context.** File, URL and context text is wrapped in `<untrusted_context>…</untrusted_context>`, and a grounding rule tells the worker to treat it as data, not instructions. This guards against prompt injection hidden in a README or web page.

`DelegateTaskResult.guardrail_actions` lists what was done on each call. These checks match patterns, so they catch the common cases but are not a guarantee. For genuinely confidential data, use `provider='ollama'`.

### Task Types (`task_type`)

`config/task_profiles.yaml` defines kinds of work: `review`, `summarize`, `extract`, `classify`, `explain`, `translate` and `generate`. Each profile can set:

- `system_addon`: an output contract added to the system prompt.
- `default_budget`, `output_format` and `json_schema`: used only when the caller does not set them.
- `validators`:
  - `json_repair`: if the JSON fails to parse or fails the schema, the same tier gets one more try with its own output and the errors (`json_repaired` on success).
  - `citations`: every `path:line` or `{file, line, evidence_quote}` in the answer is checked against the files that were in the context. The results go to `result.verification` (`ok` / `file_not_in_context` / `line_out_of_range` / `quote_not_found`), and each failed one gets a `citation_unverified:<file>:<line>` warning.

For example, `review` returns `findings: [{file, line, severity, claim, evidence_quote}]` at L3. Leaving out `task_type` gives exactly the old behaviour. To add a type, add a YAML entry; no code change is needed.

### Local RAG

If `file_paths` includes a directory or glob, `rag.enabled` is `true`, and the files' full text would not fit the budget's context window (`num_ctx` minus the 25% reserve), the server embeds them with `rag.embedding_model` via the Ollama at `rag.embedding_endpoint`. A local FAISS store then picks the `top_k` chunks most relevant to the `task`. Below that size the model gets every file verbatim, which is what review needs. If Ollama is unreachable, the call reports `rag_indexing_failed` and falls back to the full text.

### Response Caching

Responses are cached based on a hash of the final built prompt, generation options, and model. This avoids repeated inference costs for identical queries. The cache is in-memory and configurable in `config/default.yaml`:

```yaml
cache:
  enabled: true
  max_entries: 256
  ttl_seconds: 3600
```

### Background Tasks (`async_mode` & `check_job_status`)

Large tasks (Budget L3 or Map-Reduce) may exceed Claude's timeout limit. Calling `delegate_task(..., async_mode=true)` immediately returns a `job_id`. Claude can then periodically poll the `check_job_status(job_id)` tool to retrieve the results once completed.

### Map-Reduce for Large Contexts

`delegate_task(..., map_reduce=true)` automatically splits massive contexts (or large sets of files) into chunks that fit the budget's `num_ctx`. It processes each chunk concurrently (Map phase, bounded by the target provider's `concurrency`), then synthesizes the results (Reduce phase). This requires multiple LLM calls and is **opt-in only**.

## Registering with Claude Code

The server is installed as an editable package, providing the `llm-worker-pool.exe` console script. Register it with Claude Code (user scope):

```bash
claude mcp add -s user llm-worker-pool -- "D:\project\mcp\.venv\Scripts\llm-worker-pool.exe"
```

Because it's installed in editable mode, any source code changes in `src/` will take effect immediately upon server restart without needing to re-register.

## Running Tests

```bash
# Unit tests (mocks LLM, does not require Ollama)
.venv\Scripts\python.exe -m pytest tests/unit -v

# Integration tests (requires Ollama running - auto-skips if unavailable)
.venv\Scripts\python.exe -m pytest tests/integration -v
```

## Usage Stats

The server keeps counters in memory and exposes them through the `get_gateway_stats`
MCP tool — but reading them that way costs a model round-trip just to look at numbers.
Every update is therefore also mirrored, atomically, to `~/.llm_worker_pool/stats.json`,
which anything can read directly:

```powershell
worker-stats            # human-readable table
worker-stats --json     # raw snapshot, for piping into jq
```

Tracked: requests, failures, cache hits, **tokens actually spent** split into
prompt / completion / thinking, and the count of answers cut off at the output ceiling.
Cache hits are excluded from spend and counted separately as tokens saved.

Counters reset when the server restarts, since they live in the server process.

### Prometheus metrics (`worker-metrics`)

Each server instance appends one JSONL line per call (and one per failed tier) to `~/.llm_worker_pool/events.jsonl`. Every line is a single `O_APPEND` write, so several instances (Claude Code, Gemini CLI…) can share the file safely. It rotates to `events.jsonl.1` at `metrics.max_bytes`. Task and response text are never logged unless `logging.debug_log_content` is on. `stats.json` and `/worker` work the same as before.

A separate exporter process turns the log into metrics. It reads only the new bytes on each scrape:

```bash
uv pip install -e ".[metrics]"
worker-metrics                      # http://127.0.0.1:9464/metrics
curl -s 127.0.0.1:9464/metrics | grep llm_worker_
```

It is a separate process because MCP runs over stdio and often as several instances at once. An HTTP endpoint inside each instance would fight over the port, and each would only see its own part of the traffic.

| metric | labels |
|---|---|
| `llm_worker_requests_total` | provider, model, budget, task_type, outcome |
| `llm_worker_latency_seconds` (histogram) | provider, budget, outcome |
| `llm_worker_tokens_total` | provider, kind (prompt/completion/thinking) |
| `llm_worker_fallback_total` | requested, answered |
| `llm_worker_errors_total` | provider, code (timeout, http_503, prompt_too_large…) |
| `llm_worker_warnings_total` | code (fallback_used, output_truncated_at_max_tokens, json_parse_failed, secret_redacted, citation_unverified…) |
| `llm_worker_guardrail_actions_total` | action |
| `llm_worker_eval_score` (gauge) | metric, provider, budget, task_type (from `eval_latest.json`) |

`InvocationLogger(sinks=[...])` is a hook for a future tracer such as a self-hosted Langfuse. Nothing is installed for it yet.

## Evaluation (`worker-eval`)

```bash
uv pip install -e ".[eval]"
worker-eval --arm groq --arm gemini-lite --arm gemini --label baseline        # harness off
worker-eval --arm groq --arm gemini-lite --harness on --compare evals/reports/<baseline>.json
worker-eval --compare A.json B.json                                          # diff only
worker-eval --arm groq --arm gemini-lite --harness on --emit-routing         # measured routing
worker-eval --from-report NEW.json --gate BASELINE.json                      # regression gate
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
- **Routing from eval.** `--emit-routing` writes `config/routing_from_eval.yaml`. For each task type it picks the provider with the best primary metric (review: `bug_recall`, extract/classify: `field_accuracy`, summarize/explain: `faithfulness` or `must_contain`, translate: `factual_correctness`). Only providers that answered at least `--min-success` (0.8) of their cases qualify. Each pick carries its evidence as a comment. The file fills `preferred_provider` in the task profiles, and a value written in `task_profiles.yaml` wins. Review the file before committing it.
- **Regression gate.** `--gate BASELINE.json` exits 1 when an arm's primary metric for a task type drops by more than `--gate-tolerance` (0.05). A drop in `success` is only printed as a note. `--from-report` applies `--emit-routing`/`--gate` to an existing report without calling any model.
- **Output.** Reports go to `evals/reports/<timestamp>[-label].{json,md}`. Scores are also merged into `~/.llm_worker_pool/eval_latest.json` for the exporter, so a `--case`/`--task-type` subset run only replaces what it measured.

### `/worker` slash command in Claude Code

`~/.claude/commands/worker.md` runs the CLI through bash injection, so the table is
produced by the command itself rather than generated by the model:

```markdown
---
description: Xem thống kê LLM Worker Pool
allowed-tools: Bash(D:/project/mcp/.venv/Scripts/python.exe -m llm_worker_pool.cli:*)
---
!`D:/project/mcp/.venv/Scripts/python.exe -m llm_worker_pool.cli`
```

Typing `/worker` prints the current stats. The model only adds a comment when a number
warrants it — truncated answers, thinking above 40% of spend, or a provider erroring out.

## Extensibility

To **add a new provider type** (e.g., vLLM, llama.cpp):
*(Note: OpenAI, Gemini, Groq are already supported via the `openai_compatible` type)*
1. Create a class implementing the `LLMProvider` interface in `providers/`.
2. Create a builder function and register it in `providers/registry.py`.
3. Add the corresponding block to `default.yaml`.

To **add a new budget profile**: simply add an entry to `config/budgets.yaml`.
No code changes are required for either!
