# local-ai-gateway

MCP server đóng vai trò **middleware giữa Claude Code và các LLM chạy local**
(hiện tại qua [Ollama](https://ollama.com), tương lai có thể thêm vLLM,
llama.cpp, OpenAI-compatible API, Gemini, ...).

Đây **không phải chatbot**. Claude Code luôn là bộ não ra quyết định cuối
cùng; model local chỉ là "worker" chuyên biệt — dùng để nén context (đọc
nhiều file, tóm tắt, tìm bug candidate, lên kế hoạch sơ bộ) hoặc xử lý các
tác vụ tốn effort thấp/trung bình, giúp tiết kiệm context và token của
Claude.

Toàn bộ hệ thống chỉ expose **đúng một tool MCP**: `call_local`. Claude
không cần biết model nào đang chạy phía sau hay tham số sinh văn bản cụ
thể ra sao — mọi thứ nằm sau interface.

## Kiến trúc

```
Claude Code (supervisor)
      │ call_local(task, budget, context, output_format, json_schema)
      ▼
MCP layer        server.py — đăng ký tool, map input/exception
      ▼
Application layer  CallLocalUseCase — điều phối toàn bộ flow
      ▼
Domain layer      interfaces / models / PromptBuilder / OutputParser
      ▼
Provider layer    ProviderFactory → OllamaProvider (httpx → Ollama /api/chat)
```

Nguyên tắc: domain không phụ thuộc gì cả; application chỉ biết domain qua
interface `LLMProvider`; provider implement interface đó; MCP layer chỉ
wiring + validate input, không chứa logic nghiệp vụ. Thêm provider mới
hoặc budget mới **không đụng vào code hiện có** — xem mục [Mở rộng](#mở-rộng).

Xem chi tiết thiết kế/lý do từng lớp trong lịch sử trao đổi khi kiến trúc
này được duyệt (không có file thiết kế riêng trong repo).

## Cấu trúc thư mục

```
mcp/
├── pyproject.toml
├── .env.example
├── config/
│   ├── default.yaml        # provider, endpoint, timeout, retry, logging
│   └── budgets.yaml        # định nghĩa budget L1/L2/L3
├── src/local_ai_gateway/
│   ├── server.py            # MCP entrypoint, đăng ký tool call_local
│   ├── application/
│   │   ├── dto.py                    # CallLocalRequest / CallLocalResult
│   │   └── call_local_use_case.py    # điều phối flow chính
│   ├── domain/
│   │   ├── interfaces.py     # LLMProvider (Protocol)
│   │   ├── models.py         # GenerationOptions, ProviderResponse, BudgetProfile
│   │   ├── exceptions.py
│   │   ├── prompt_builder.py # ghép task+context, cắt context nếu vượt num_ctx
│   │   └── output_parser.py  # parse/repair JSON, validate schema optional
│   ├── providers/
│   │   ├── registry.py       # ProviderFactory
│   │   └── ollama_provider.py
│   ├── config/
│   │   └── settings.py       # AppConfig (Pydantic Settings) + load budgets.yaml
│   └── infra/
│       ├── logging.py        # structured logger, ghi stderr (không phá stdio)
│       └── retry.py          # backoff cho lỗi transient
└── tests/
    ├── unit/         # mock LLMProvider, không cần Ollama chạy
    └── integration/  # gọi Ollama thật, tự skip nếu Ollama không chạy
```

## Yêu cầu

- Python ≥ 3.11
- [Ollama](https://ollama.com) đang chạy (`ollama serve`, mặc định
  `http://localhost:11434`) với model đã pull sẵn (mặc định `qwen3.5:9b`):
  ```
  ollama pull qwen3.5:9b
  ```

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Cấu hình

Mọi tham số đều config-driven, **không cần sửa code**:

- `config/default.yaml` — provider, endpoint Ollama, model, timeout, retry,
  concurrency, logging.
- `config/budgets.yaml` — định nghĩa các execution budget (xem bên dưới).
  Thêm budget mới = thêm entry vào file này.

Override bằng biến môi trường (prefix `LOCAL_AI_GATEWAY_`, các trường lồng
nhau nối bằng `__`) hoặc bằng file `.env` ở gốc repo — copy từ
`.env.example`. Thứ tự ưu tiên (cao → thấp):

```
init kwargs (code)  >  biến môi trường thật  >  file .env  >  default.yaml
```

Ví dụ đổi model không cần sửa file:

```bash
set LOCAL_AI_GATEWAY_OLLAMA__MODEL=qwen3-coder:8b
```

### Budget profiles

| Budget | Mục đích | thinking | num_ctx | num_predict | timeout |
|--------|----------|----------|---------|-------------|---------|
| `L1` | Nhanh, deterministic — phân loại, trích xuất, trả lời ngắn | tắt | 4096 | 256 | 20s |
| `L2` | Cân bằng — tóm tắt, refactor vừa phải, giải thích code | mặc định | 12288 | 1024 | 60s |
| `L3` | Chất lượng cao nhất — phân tích kiến trúc, lên kế hoạch, đọc nhiều file | mặc định | 24576 | 4096 | 180s |

> Lưu ý: `qwen3.5:9b` là model có khả năng "thinking". Nếu bật thinking mà
> `num_predict` quá nhỏ, model có thể tiêu hết token cho phần suy nghĩ nội
> bộ và không còn token nào cho câu trả lời cuối. Vì vậy `L1` tắt hẳn
> thinking (`think: false`) để đảm bảo luôn có câu trả lời trong ngân sách
> token thấp.

Giá trị `num_ctx` được chọn ở mức an toàn cho VRAM 12GB (weights ~6.6GB +
KV cache), không dùng tới context tối đa 262144 mà model hỗ trợ — có thể
tune lại trong `config/budgets.yaml` nếu đo được còn dư/thiếu VRAM.

## Đăng ký vào Claude Code

Server đã cài dạng editable nên có sẵn console script
`.venv\Scripts\local-ai-gateway.exe`. Đăng ký (user scope — dùng được ở
mọi project):

```bash
claude mcp add -s user local-ai-gateway -- "D:\project\mcp\.venv\Scripts\local-ai-gateway.exe"
```

Kiểm tra:

```bash
claude mcp get local-ai-gateway
```

Gỡ đăng ký:

```bash
claude mcp remove local-ai-gateway -s user
```

Vì cài ở chế độ editable, sửa code trong `src/` không cần đăng ký lại —
Claude Code sẽ chạy code mới nhất mỗi lần khởi động server.

## Sử dụng — tool `call_local`

```
call_local(
    task: str,                 # bắt buộc — mô tả việc cần làm
    budget: str = "L2",        # "L1" | "L2" | "L3" (hoặc budget tự thêm)
    context: str | None,       # nội dung/code cần model đọc
    output_format: "text" | "json" = "text",
    json_schema: dict | None,  # gợi ý cấu trúc JSON mong muốn (optional)
)
```

Kết quả trả về (`CallLocalResult`):

```json
{
  "success": true,
  "content": "...",
  "raw_text": "...",
  "provider": "ollama",
  "model": "qwen3.5:9b",
  "budget_used": "L2",
  "latency_ms": 1234.5,
  "tokens_used": 256,
  "warnings": [],
  "error": null
}
```

Khi có lỗi (timeout, Ollama không kết nối được, JSON không parse được...),
`success=false` kèm `error`/`warnings` mô tả rõ nguyên nhân — tool **không
throw lỗi cứng cho các lỗi vận hành**, để Claude tự đọc và quyết định
(thử lại, hạ budget, hỏi lại user...). Chỉ lỗi cấu hình (budget không tồn
tại, provider chưa đăng ký...) mới trả về MCP tool error.

## Chạy test

```bash
# Unit test (không cần Ollama)
.venv\Scripts\python.exe -m pytest tests/unit -v

# Integration test (cần Ollama đang chạy — tự skip nếu không kết nối được)
.venv\Scripts\python.exe -m pytest tests/integration -v
```

## Mở rộng

**Thêm provider mới** (vLLM, OpenAI, Gemini, llama.cpp, ...):
1. Viết class implement `LLMProvider` (`domain/interfaces.py`) trong
   `providers/`.
2. Đăng ký trong `providers/registry.py` (`_REGISTRY` hoặc gọi
   `register_provider(...)`).
3. Thêm block config cho provider đó trong `default.yaml`.

Không cần sửa `application/`, `domain/`, hay `server.py`.

**Thêm budget mới**: thêm entry vào `config/budgets.yaml` — không sửa code.

## Ghi chú thiết kế

- Server dùng transport **stdio** — mọi log ghi ra **stderr**
  (`infra/logging.py`), không bao giờ ghi ra stdout vì stdout dành riêng
  cho giao thức MCP.
- Concurrency mặc định giới hạn 1 request gọi provider cùng lúc
  (`concurrency.max_parallel_requests`) để tránh tràn VRAM 12GB khi chạy
  nhiều request context lớn song song.
- Log invocation mặc định **không ghi nội dung** task/context/response,
  chỉ ghi độ dài — bật `logging.debug_log_content=true` khi cần debug local.
