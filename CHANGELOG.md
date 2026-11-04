# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-11-03

### Changed
- The project is renamed from `llm-worker-pool` to Copro: package `copro-mcp`, module
  `copro`, MCP server name `copro`, environment prefix `COPRO_`, user directory
  `~/.copro`, Prometheus metrics `copro_*`.
- One `copro` command with `serve`, `stats`, `metrics` and `eval` subcommands, plus
  `copro-mcp` to start the server (`uvx copro-mcp`).
- `copro stats` prints in English.
- Mistral, DeepSeek and OpenRouter are marked experimental until they have been run
  against the live APIs.

### Deprecated
- The old names keep working until 0.4.x ends, each with a note on stderr:
  `LLM_WORKER_POOL_*` variables, `~/.llm_worker_pool` (copied to `~/.copro` on the
  first start) and the `llm-worker-pool`, `worker-stats`, `worker-metrics` and
  `worker-eval` commands.

## [0.3.0] - 2026-11-02

### Added
- Mistral, DeepSeek and OpenRouter providers, each skipped until its key is set and
  already part of every fallback chain. Per-provider `send_reasoning_effort`,
  `structured_output` and `extra_body` adapt the OpenAI-compatible adapter to each API.
- Install with `uv tool install` / `uvx`: the YAML config ships inside the package;
  `~/.llm_worker_pool/.env` holds keys and settings for an installed tool.
- Task types (`task_type`): an output contract per kind of work, a `verification`
  field checking every cited `path:line`, one JSON repair retry.
- Cross-check reviews: a review runs on two vendors in parallel and the findings are
  merged, each marked with who reported it.
- Guardrails: a deny-list for secret files, redaction of secret-looking literals
  before any cloud call, untrusted-context fencing.
- `continuation_id` follow-ups that reuse the earlier context without resending files.
- `file_paths` accepts directories (walked recursively) and globs.
- Quota-aware routing: saturated tiers are skipped before the call; a 429 is retried
  after the time the key is actually locked, or the chain moves on.
- Key quota and circuit-breaker state shared by every server process on the machine.
- Fallback tiers whose prompt cap the request exceeds are dropped up front; content-aware
  token estimates for those caps.
- Single-provider reviews over a provider's prompt cap are split into batches of whole files.
- MCP progress notifications for long calls, and the latest step in async job status.
- Token and thinking accounting, truncation warnings, `worker-stats` and `/worker`,
  per-key circuit-breaker state in `get_gateway_stats`.
- `worker-eval` (RAGAS plus deterministic metrics) with `--repeat`, routing emitted
  from measurements (`--emit-routing`, `--switch-margin`) and a regression gate (`--gate`).
- Event log and Prometheus exporter (`worker-metrics`).
- URL context (`urls`), including shallow-cloned GitHub repositories.

### Changed
- Budgets L1-L3 route to Groq first, with Gemini lite before flash on L3.
- `default_provider` is `gemini-lite`.
- `file_access.allowed_roots` ships empty: each machine names its own roots.
- RAG is off by default; faiss and numpy moved to the optional `[rag]` extra.
- Groq also reads `GROQ_API_KEY`, so one key per vendor is enough to start.
- Eval reports mask provider account ids.

### Fixed
- The shared key-state lock no longer blocks the event loop for a second at a time on Windows.
- Follow-ups keep the settings the server passes explicitly as `None`.
- Quoted evidence that a worker escaped one time too many is matched.
- Non-UTF-8 text files are sent (with a `file_decoded_lossy` warning) instead of dropped.
- A failed call is no longer scored as zero quality in evals.

## [0.2.0] - 2026-08-09

### Added
- Multiple providers (Gemini through the OpenAI-compatible adapter), map-reduce for
  large contexts, async jobs, local RAG, usage stats and a file cache.
- Per-provider concurrency limits and multi-key round-robin.
- Routing by confidentiality: cloud by default, Ollama for content that must stay local.

### Changed
- The package is renamed from `local_ai_gateway` to `llm_worker_pool`.

### Fixed
- Gemini responses that carry `extra_content` with thinking state are parsed.
- `.env` is loaded into the real process environment, so provider key variables set
  there are found.

## [0.1.0] - 2026-08-07

### Added
- First MCP server (`local-ai-gateway`): budget profiles, graceful degradation,
  best-effort JSON parsing, concurrency guards, `file_paths` and a response cache.
