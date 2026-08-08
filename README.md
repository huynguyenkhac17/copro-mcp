# LLM Worker Pool

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-1.2.0+-green.svg)](https://modelcontextprotocol.io/)

An MCP (Model Context Protocol) server acting as a middleware between Claude Code and a pool of local/cloud LLM workers.

This project is **not a chatbot**. Claude Code remains the ultimate decision-making supervisor; the local models act as specialized "workers" used to compress context (reading multiple files, summarizing, finding bug candidates, initial planning) or handle low/medium effort tasks. This significantly saves Claude's context window and API token usage.

The entire system exposes exactly **one MCP tool**: `call_local`. Claude does not need to know which model is running in the background or what specific text generation parameters are used—everything is abstracted behind a clean interface.

## Architecture

```
Claude Code (supervisor)
      │ call_local(task, budget, context, output_format, json_schema)
      ▼
MCP layer        server.py — registers the tool, maps inputs/exceptions
      ▼
Application layer  CallLocalUseCase — orchestrates the main flow
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
├── src/local_ai_gateway/
│   ├── server.py           # MCP entrypoint, registers call_local tool
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
- [Ollama](https://ollama.com) running locally (`ollama serve`, default `http://localhost:11434`) with a pulled model (default `qwen3.5:9b`):
  ```bash
  ollama pull qwen3.5:9b
  ```
- *(Optional)* To use the `gemini` provider out-of-the-box, set the `GEMINI_API_KEY` environment variable. If omitted, the gemini provider is **skipped with a warning**, and the server will continue running normally with local models.
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

You can override settings via environment variables (prefix `LOCAL_AI_GATEWAY_`, use `__` for nested fields) or via a `.env` file at the repository root. Priority (highest to lowest):
`init kwargs > real environment variables > .env file > default.yaml`

### Providers

`config/default.yaml` declares multiple providers simultaneously. Each provider has a name and a `type` (the implementation class in `providers/registry.py`):

```yaml
default_provider: ollama
providers:
  ollama:
    type: ollama
    endpoint: http://localhost:11434
    model: qwen3.5:9b
  gemini:
    type: openai_compatible   # Compatible with Gemini/OpenAI/Groq
    endpoint: https://generativelanguage.googleapis.com/v1beta/openai
    model: gemini-2.5-flash
    api_key_env: GEMINI_API_KEY   # The ENV VAR name containing the key
```

The `call_local` tool accepts a `provider: str | None` parameter to select the backend per invocation (defaults to `default_provider`). This allows routing heavy reasoning to `gemini` while keeping lightweight tasks on `ollama`.

### Budget Profiles

| Budget | Purpose | thinking | num_ctx | num_predict | timeout |
|--------|---------|----------|---------|-------------|---------|
| `L1` | Fast, deterministic — categorization, extraction, short answers | disabled | 4096 | 256 | 20s |
| `L2` | Balanced — summarizing, moderate refactoring, code explanation | disabled | 12288 | 1024 | 60s |
| `L3` | Highest quality — architecture analysis, bug finding, reading multiple files | default | 24576 | 4096 | 180s |

> Note: For models with "thinking" capabilities (like `qwen3.5`), enabling thinking with a small `num_predict` may cause the model to consume all tokens on internal reasoning, returning an empty response. Thus, `think: false` is enforced for L1 and L2 budgets to guarantee output.

### Direct File Access (`file_paths`)

Instead of Claude reading files and pasting them into the prompt (wasting tokens), the server can read files directly via absolute paths. This is **disabled by default** for safety. Configure the allowed directories in `config/default.yaml`:

```yaml
file_access:
  allowed_roots: ["D:/project/mcp"]   # Empty = disabled
  max_file_bytes: 200000              # Limit per file
  max_total_bytes: 800000             # Limit per invocation
```

Errors (path traversal, file not found, too large, unreadable) are gracefully handled and appended to `CallLocalResult.warnings`.

### Local RAG

If `file_paths` points to a directory and `rag.enabled` is `true`, the server will embed the directory contents using a local embedding model (via Ollama) and use a local FAISS vector store to inject only the most relevant text chunks into the context based on the current `task`. This avoids flooding the context window with entire large directories.

### Response Caching

Responses are cached based on a hash of the final built prompt, generation options, and model. This avoids repeated inference costs for identical queries. The cache is in-memory and configurable in `config/default.yaml`:

```yaml
cache:
  enabled: true
  max_entries: 256
  ttl_seconds: 3600
```

### Background Tasks (`async_mode` & `check_job_status`)

Large tasks (Budget L3 or Map-Reduce) may exceed Claude's timeout limit. Calling `call_local(..., async_mode=true)` immediately returns a `job_id`. Claude can then periodically poll the `check_job_status(job_id)` tool to retrieve the results once completed.

### Map-Reduce for Large Contexts

`call_local(..., map_reduce=true)` automatically splits massive contexts (or large sets of files) into chunks that fit the budget's `num_ctx`. It processes each chunk concurrently (Map phase), then synthesizes the results (Reduce phase). This requires multiple LLM calls and is **opt-in only**. 

## Registering with Claude Code

The server is installed as an editable package, providing the `local-ai-gateway.exe` console script. Register it with Claude Code (user scope):

```bash
claude mcp add -s user local-ai-gateway -- "D:\project\mcp\.venv\Scripts\local-ai-gateway.exe"
```

Because it's installed in editable mode, any source code changes in `src/` will take effect immediately upon server restart without needing to re-register.

## Running Tests

```bash
# Unit tests (mocks LLM, does not require Ollama)
.venv\Scripts\python.exe -m pytest tests/unit -v

# Integration tests (requires Ollama running - auto-skips if unavailable)
.venv\Scripts\python.exe -m pytest tests/integration -v
```

## Extensibility

To **add a new provider type** (e.g., vLLM, llama.cpp):
*(Note: OpenAI, Gemini, Groq are already supported via the `openai_compatible` type)*
1. Create a class implementing the `LLMProvider` interface in `providers/`.
2. Create a builder function and register it in `providers/registry.py`.
3. Add the corresponding block to `default.yaml`.

To **add a new budget profile**: simply add an entry to `config/budgets.yaml`.
No code changes are required for either!
