# Rosetta Research Agent

Multi-language AI financial research agent for enterprise deployment. Generates structured, verifiable investment theses across 5 regional desks via a clean REST API.

[![CI](https://github.com/Mihai-Codes/rosetta-research-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Mihai-Codes/rosetta-research-agent/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![AdalFlow](https://img.shields.io/badge/AdalFlow-SylphAI-purple)](https://github.com/SylphAI-Inc/AdalFlow)
[![License: Proprietary](https://img.shields.io/badge/license-proprietary-red)](./LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](./Dockerfile)

## What This Is

A productized enterprise API extracted from [Rosetta Alpha](https://github.com/Mihai-Codes/rosetta-alpha) (Canteen x Arc Agora Agents hackathon, May 2026). Five regional agents each reason in their native language using specialized frontier models, then return a structured `InvestmentThesis` JSON with direction, confidence, reasoning chains, and optional IPFS/on-chain verification.

**This repo is the enterprise product. The hackathon demo lives at [rosetta-alpha.vercel.app](https://rosetta-alpha.vercel.app).**

## Quick Start

```bash
# Minimum viable (US + Crypto desks only)
export GROQ_API_KEY=your_key_here
docker compose up --build

# Full 5-desk deployment
export GROQ_API_KEY=xxx
export DEEPSEEK_API_KEY=xxx
export GOOGLE_API_KEY=xxx
docker compose up --build
```

API is live at `http://localhost:8000`. Try:

```bash
# Health check
curl http://localhost:8000/health

# List available desks
curl http://localhost:8000/api/v1/desks

# Generate a thesis
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "desk": "us"}'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with feature flags |
| GET | `/api/v1/desks` | List all desks and their capabilities |
| POST | `/api/v1/analyze` | Generate investment thesis |
| GET | `/api/v1/thesis/{id}` | Retrieve a previously generated thesis |

### POST /api/v1/analyze

**Request:**
```json
{
  "ticker": "AAPL",
  "desk": "us",
  "language_override": null
}
```

**Response:** See [AGENT.md](./AGENT.md) for full sample output.

## Deploy as Enterprise Agent

### Docker (recommended)

```bash
# 1. Clone
git clone https://github.com/Mihai-Codes/rosetta-research-agent.git
cd rosetta-research-agent

# 2. Configure
cp .env.example .env
# Edit .env with your API keys

# 3. Run
docker compose up --build -d

# 4. Verify
curl http://localhost:8000/health
```

### Without Docker

```bash
pip install -e ".[full]"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Enables US + Crypto desks |
| `DEEPSEEK_API_KEY` | No | Enables China desk (native Chinese reasoning) |
| `GOOGLE_API_KEY` | No | Enables EU + Japan desks (Gemini) |
| `TUSHARE_TOKEN` | No | Premium A-share data for China desk |
| `FINANCIAL_DATASETS_API_KEY` | No | SEC filings for US desk |
| `ENABLE_IPFS` | No | Set to "true" to pin traces to IPFS |
| `ENABLE_ONCHAIN` | No | Set to "true" for Arc L1 recording |
| `PINATA_JWT` | No | Required if ENABLE_IPFS=true |
| `STORACHA_SIDECAR_URL` | No | Storacha/Filecoin pinning (default: localhost:3030) |

## Architecture

```
POST /api/v1/analyze {"ticker": "AAPL", "desk": "us"}
        |
        v
  [Agent Factory] --> USAgent / ChinaAgent / EUAgent / JapanAgent / CryptoAgent
        |
        v
  [Sub-Agent Pipeline: Fundamental -> Technical -> Sentiment]
        |
        v
  [Thesis Synthesis (AdalFlow Generator)]
        |
        v
  [Optional: IPFS Pin (MultiPinner) + Arc L1 Record]
        |
        v
  AnalyzeResponse JSON
```

## Regional Desks

| Desk | Model | Reasoning Language | Data Sources |
|------|-------|--------------------|--------------|
| US | Groq Llama-3.3-70B | English | yfinance, Financial Datasets, Stooq |
| China | DeepSeek V4 Pro | Simplified Chinese | AKShare, Tushare Pro |
| EU | Gemini 2.5 Flash | English | yfinance, Stooq |
| Japan | Gemini 2.5 Flash | Japanese | yfinance, Stooq |
| Crypto | Groq Llama-3.3-70B | English | CoinGecko, DeFiLlama |

## Agentalent.ai Listing

This agent is listed on [Agentalent.ai](https://agentalent.ai) (monday.com's managed AI agent marketplace). See [AGENT.md](./AGENT.md) for the full capability description, pricing justification, SLA, and sample output.

## Prior Work

This is a productized extraction from the hackathon project:
- **Source repo:** [github.com/Mihai-Codes/rosetta-alpha](https://github.com/Mihai-Codes/rosetta-alpha)
- **Live demo:** [rosetta-alpha.vercel.app](https://rosetta-alpha.vercel.app)
- **Hackathon:** Canteen x Arc Agora Agents (May 11-25, 2026)
- **Post-optimization score:** ~9.0/10 composite judge

## License

Proprietary. Contact [hello@mihai.codes](mailto:hello@mihai.codes) for enterprise licensing.
