# AGENTS.md

This file provides guidance to agents (i.e., ADAL) when working with code in this repository.

---

## Essential Commands

### Python (uv — always use `uv run`, never bare `python`/`pytest`)

```bash
# Install all deps including dev
uv sync --extra dev

# Run all tests
uv run pytest tests/ -v --tb=short

# Run a single test
uv run pytest tests/test_api.py::test_analyze_success_mocked -v

# Lint (must pass — CI fails on errors)
uv run ruff check src/ tests/

# Format check (must pass — CI fails)
uv run ruff format --check src/ tests/

# Auto-fix lint + format
uv run ruff check --fix src/ tests/ && uv run ruff format src/ tests/

# Start dev server
uv run uvicorn src.api.main:app --reload --port 8000
```

**Gotcha:** `ruff format --check` is a separate CI step from `ruff check`. Both must pass. Always run both before committing.

**Gotcha:** The `[tool.ruff.lint] ignore = ["E402"]` suppression in `pyproject.toml` is intentional — every agent/data module calls `load_dotenv(override=True)` before imports so API keys are available at module init time. Do not remove this or restructure those imports.

### Docker

```bash
docker build -t rosetta-research-agent:ci .
docker run -d --name rra \
  -e GROQ_API_KEY=your_key \
  -e ENABLE_IPFS=false \
  -p 8000:8000 \
  rosetta-research-agent:ci
curl http://localhost:8000/health
```

### Quick API smoke test (server must be running)

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:8000/api/v1/desks | python3 -m json.tool
curl -s -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "desk": "us"}' | python3 -m json.tool
```

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | **Yes** (US + Crypto) | Llama-3.3-70B for US/Crypto desks |
| `DEEPSEEK_API_KEY` | No | China desk (DeepSeek V4 Pro, native Chinese) |
| `GOOGLE_API_KEY` | No | EU + Japan desks (Gemini 2.5 Flash) |
| `TUSHARE_TOKEN` | No | Premium A-share data for China desk |
| `FINANCIAL_DATASETS_API_KEY` | No | SEC filings for US desk |
| `ENABLE_IPFS` | No | Set `"true"` to pin traces |
| `PINATA_JWT` | No | Required if `ENABLE_IPFS=true` |
| `STORACHA_SIDECAR_URL` | No | Default `http://localhost:3030` |
| `ENABLE_ONCHAIN` | No | Arc L1 recording |

CI runs with `GROQ_API_KEY=test-key-placeholder` and `ENABLE_IPFS=false` — all LLM calls are mocked in tests so no real key is needed.

---

## Architecture: Request → InvestmentThesis

```
POST /api/v1/analyze {"ticker": "AAPL", "desk": "us"}
        │
        ▼
  AnalyzeRequest (Pydantic, extra="forbid")
  • ticker: allowlist regex [A-Za-z0-9._/-]{1,24}
  • desk: must be in VALID_DESKS
  • timeout_seconds: ge=15.0, le=600.0
        │
        ▼
  _get_agent(desk) → lazy-import USAgent / ChinaAgent / EUAgent / JapanAgent / CryptoAgent
        │
        ▼
  agent.analyze(ticker)          ← asyncio.wait_for() with timeout
  ┌─────────────────────────────────────────────────────┐
  │  RegionalAgent.analyze() [base_agent.py]            │
  │  1. get_data_sources(ticker) → raw market data dict │
  │  2. For each sub_agent_role in sub_agent_roles:      │
  │     • Build prompt via SUB_AGENT_TEMPLATE            │
  │     • adal.Generator call (LLM)                      │
  │     • PydanticJsonParser → ReasoningBlock            │
  │     • On malformed JSON: one-shot json_repair pass   │
  │  3. Build synthesis prompt (all ReasoningBlocks)     │
  │  4. adal.Generator call → InvestmentThesis JSON      │
  │  5. PydanticJsonParser strips unknown fields FIRST   │
  │     then validates (guards against LLM hallucination)│
  └─────────────────────────────────────────────────────┘
        │
        ▼
  Optional: MultiPinner.pin() [persistence/multi_pinner.py]
  • Serializes with orjson OPT_SORT_KEYS (deterministic bytes → same CID)
  • Uploads in parallel to Pinata + Storacha
  • Quorum: require=2 — both must succeed
  • Asserts both return identical CID (content-addressing invariant)
        │
        ▼
  AnalyzeResponse → stored in _thesis_store[thesis_id] for GET /thesis/{id}
```

---

## Non-Obvious Relationships

### PydanticJsonParser — why it exists
AdalFlow's built-in `JsonOutputParser` requires `adal.DataClass`. Domain models are Pydantic `BaseModel` (FastAPI + web3 interop). `PydanticJsonParser` in `base_agent.py` bridges this: it strips LLM-hallucinated extra fields before Pydantic validation so `extra="forbid"` doesn't crash on novel LLM outputs. **Never bypass this parser or feed raw LLM output directly to Pydantic models.**

### JSON repair flow
If the LLM returns malformed JSON (truncated, trailing commas, etc.), `base_agent.py` runs a deterministic one-shot repair pass before the Pydantic parse. This is intentional and tested — do not remove it. The repair uses `json_repair` library (imported lazily inside the method).

### load_dotenv placement
All agent and data module files call `load_dotenv(override=True)` at module top, **before** `import adalflow as adal`. This is required: AdalFlow's model clients read API keys at import time. Ruff E402 is suppressed for exactly this pattern.

### model_name on agents
`AnalyzeResponse.model_used` is populated via `getattr(agent, "model_name", None)` with an `isinstance(…, str)` guard. In tests, the agent is a `MagicMock` — without the guard `model_used` would receive a `MagicMock` object and fail `AnalyzeResponse` validation.

### _thesis_store
In-memory dict keyed by `thesis_id` (UUID). Not persisted across restarts. Swap for Redis/Postgres in production. Used only by `GET /api/v1/thesis/{id}`.

### Sub-agent roles per desk
Each `RegionalAgent` subclass declares `sub_agent_roles` as a class-level tuple. The base class iterates them sequentially (not in parallel) to maintain reasoning chain coherence. Adding a new role = add to the tuple + handle the role's prompt in `build_synthesis_prompt`.

---

## Key Entry Points

| What | Where |
|---|---|
| FastAPI app + all endpoints | `src/api/main.py` |
| Base agent logic, JSON repair, sanitizers | `src/agents/base_agent.py` |
| US desk (reference implementation) | `src/agents/us_agent.py` |
| All domain schemas (InvestmentThesis, ReasoningBlock, etc.) | `src/reasoning/trace_schema.py` |
| IPFS dual-provider pinning | `src/persistence/multi_pinner.py` |
| All unit tests | `tests/test_api.py` |
| Dependencies + ruff config | `pyproject.toml` |
| Agentalent.ai agent profile | `docs/agentalent_profile.md` |

---

## Testing Strategy

- All 22 tests run **without real API keys** — LLM calls are mocked via `unittest.mock.AsyncMock`
- `_get_agent` is patched at `src.api.main._get_agent` (not at the agent module level)
- Mock agent must be a `MagicMock` (not `AsyncMock`) with `.analyze = AsyncMock(return_value=mock_thesis)`
- `mock_thesis` must be a real `InvestmentThesis` Pydantic instance (not a dict) — `AnalyzeResponse` calls `.direction.value`, `.confidence_score`, `.reasoning_blocks`, `.timestamp.isoformat()`
- The `AsyncMock` coroutine warning in test output is a pytest-asyncio artefact, not a real failure

---

## CI Pipeline (GitHub Actions)

Two jobs — both must pass:

1. **test** (matrix: Python 3.12 + 3.13)
   - `uv sync --extra dev`
   - `ruff check src/ tests/` — **hard fail**
   - `ruff format --check src/ tests/` — **hard fail**
   - `pytest -v --tb=short tests/`

2. **docker** (needs: test)
   - `docker build -t rosetta-research-agent:ci .`
   - Container smoke test: starts container, `sleep 8`, `curl --fail http://localhost:8000/health`

**Common CI failure causes:**
- Formatting drift → run `uv run ruff format src/ tests/` locally before push
- New unused import → `uv run ruff check --fix src/ tests/`
- Docker: Dockerfile uses `pip install -e ".[full]"` — if a new dep in `[full]` extras has a build dependency, add it to the `RUN apt-get` step
