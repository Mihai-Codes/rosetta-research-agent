# Rosetta Research Agent

## Agent Name
Rosetta Research Agent

## Capability Summary
Multi-language AI financial research agent that generates structured, verifiable investment theses across 5 regional desks (US, China, EU, Japan, Crypto). Each desk reasons in its native language using specialized frontier models, producing institutional-grade analysis with confidence scoring, reasoning traces, and optional on-chain audit trails.

## What It Does
- Generates structured investment theses (LONG/SHORT/NEUTRAL) with confidence scores (0-1)
- Runs multi-sub-agent analysis per desk: Fundamental, Technical, and Sentiment specialists
- Reasons natively in Chinese (DeepSeek V4 Pro) for A-shares, Japanese for Tokyo equities
- Produces full reasoning trace chains with thought_process and thesis_summary in English
- Optionally pins all reasoning traces to IPFS (Pinata + Storacha/Filecoin) for verifiability
- Optionally records trace hashes on-chain (Arc L1) for immutable audit trails
- Returns structured JSON via a clean REST API (FastAPI, versioned at /api/v1/)

## Supported Desks and Tickers

| Desk | Model | Language | Example Tickers |
|------|-------|----------|-----------------|
| US Equities | Groq Llama-3.3-70B | English | AAPL, MSFT, NVDA, TSLA, AMZN |
| China A-Shares | DeepSeek V4 Pro | Simplified Chinese | 600519.SH, 000858.SZ, 601318.SH |
| EU Equities | Gemini 2.5 Flash | English | MC.PA, SAP.DE, ASML.AS, LVMH.PA |
| Japan Equities | Gemini 2.5 Flash | Japanese | 7203.T, 6758.T, 9984.T, 8306.T |
| Crypto | Groq Llama-3.3-70B | English | BTC, ETH, SOL, AVAX |

## Required Integrations
- **GROQ_API_KEY** (minimum viable): Enables US Equities and Crypto desks
- Internet access for real-time market data (yfinance, AKShare, CoinGecko)

## Optional Integrations
- **DEEPSEEK_API_KEY**: Enables China desk with native Mandarin reasoning
- **GOOGLE_API_KEY**: Enables EU and Japan desks with Gemini models
- **TUSHARE_TOKEN**: Higher-fidelity A-share data for China desk
- **FINANCIAL_DATASETS_API_KEY**: SEC filing data for US desk
- **PINATA_JWT** + **STORACHA_SIDECAR_URL**: IPFS persistence (verifiable traces)
- **ARC_RPC_URL** + **ARC_DEPLOYER_PRIVATE_KEY**: On-chain audit trail

## Uptime SLA
99.5% monthly uptime (agent-side). Actual availability depends on upstream LLM provider status (Groq, DeepSeek, Google). Graceful degradation: if a desk's primary model is unavailable, the agent returns a clear error with status code rather than hallucinating.

## Response Time
- Typical: 15-45 seconds per thesis (includes real-time data fetch + multi-agent reasoning)
- P95: under 60 seconds
- Health check: under 100ms

## Pricing Recommendation
**$3,500/month** per enterprise seat (unlimited queries across all desks).

Justification:
- Bloomberg Terminal: $2,000-2,500/month (data only, no AI reasoning)
- Refinitiv Eikon: $1,500-2,200/month (data + basic analytics)
- AlphaSense: $1,000-2,000/month (search + summaries)
- This agent provides: structured directional calls + confidence scoring + multi-language native reasoning + verifiable audit trail. The combination of actionable output (not just data), multi-region coverage with native-language insight, and optional on-chain accountability justifies a premium above pure data terminals.

For Agentalent listing, recommend starting at **$3,500/month** to position between Bloomberg and bespoke quant research services. Volume discounts for multi-seat enterprise: 3+ seats at $3,000/month each.

## Data Sources
- yfinance (real-time prices, fundamentals)
- AKShare (China A-share OHLCV, financial indicators)
- Tushare Pro (premium A-share data, optional)
- Financial Datasets API (SEC filings, US equities)
- CoinGecko (crypto prices, market cap, volume)
- DeFiLlama (DeFi TVL, protocol analytics)
- Stooq (fallback global equities data)

## Limitations
- Not a registered investment advisor. Output is research, not financial advice.
- Real-time data has 15-minute delay on some exchanges (standard for free-tier feeds).
- China desk requires DeepSeek API key for native Chinese reasoning; falls back to English if unavailable.
- Crypto desk does not cover derivatives or perpetual futures analytics.
- Maximum one thesis per ticker per request (no batch endpoint yet).
- In-memory thesis store in current version; requires Redis/Postgres for production persistence.

## Sample Output

```json
{
  "thesis_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "ticker": "AAPL",
  "desk": "us",
  "direction": "LONG",
  "confidence": 0.78,
  "thesis_summary_en": "Apple's services revenue growth (+14% YoY) and expanding margins in wearables offset near-term iPhone cycle softness. Valuation at 28x forward P/E is justified by the recurring revenue mix shift. Entry below $195 with 12-month target $230.",
  "reasoning_blocks": [
    {
      "agent_role": "fundamental",
      "thought_process": "Services segment now 22% of revenue at 71% gross margin...",
      "conclusion": "LONG",
      "confidence": 0.82
    },
    {
      "agent_role": "technical",
      "thought_process": "Price holding 200-day MA, RSI at 55 (neutral-bullish)...",
      "conclusion": "LONG",
      "confidence": 0.71
    },
    {
      "agent_role": "sentiment",
      "thought_process": "Analyst consensus 78% Buy, institutional accumulation visible...",
      "conclusion": "LONG",
      "confidence": 0.80
    }
  ],
  "timestamp": "2026-05-28T14:30:00Z",
  "model_used": "groq/llama-3.3-70b-versatile",
  "ipfs_cid": "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
  "processing_time_ms": 23450
}
```

## Framework
Custom multi-agent system built on AdalFlow (SylphAI). Not LangChain, not CrewAI. The AdalFlow framework treats prompts as optimizable parameters via textual gradient descent, allowing continuous improvement of reasoning quality without retraining models.

## Proof of Prior Work
- Hackathon project: [github.com/Mihai-Codes/rosetta-alpha](https://github.com/Mihai-Codes/rosetta-alpha)
- Live demo: [rosetta-alpha.vercel.app](https://rosetta-alpha.vercel.app)
- Post-optimization composite judge score: ~9.0/10
- Built in 14 days, solo, for Canteen x Arc Agora Agents hackathon (May 2026)
