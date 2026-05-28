"""Rosetta Research Agent - Enterprise API.

FastAPI application providing structured investment thesis generation
across 5 regional desks (US, China, EU, Japan, Crypto).

Endpoints:
    GET  /health              - Health check
    GET  /api/v1/desks        - List available desks and their capabilities
    POST /api/v1/analyze      - Generate an investment thesis for a ticker
    GET  /api/v1/thesis/{id}  - Retrieve a previously generated thesis by ID
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory thesis store (swap for Redis/Postgres in production)
# ---------------------------------------------------------------------------

_thesis_store: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Rosetta Research Agent starting up...")
    yield
    logger.info("Rosetta Research Agent shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Rosetta Research Agent",
    version="1.0.0",
    description=(
        "Multi-language AI financial research agent. "
        "Generates structured investment theses across 5 regional desks."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

VALID_DESKS = {"us", "china", "eu", "japan", "crypto"}


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., description="Asset ticker symbol (e.g. AAPL, BTC, 600519.SH)")
    desk: str = Field(..., description="Regional desk: us, china, eu, japan, crypto")
    language_override: str | None = Field(
        None, description="Override the reasoning language (e.g. 'en', 'zh', 'ja')"
    )


class AnalyzeResponse(BaseModel):
    thesis_id: str
    ticker: str
    desk: str
    direction: str
    confidence: float
    thesis_summary_en: str
    reasoning_blocks: list[dict[str, Any]]
    timestamp: str
    model_used: str
    ipfs_cid: str | None = None
    processing_time_ms: int


class DeskInfo(BaseModel):
    name: str
    region: str
    model: str
    language: str
    supported_tickers: str
    data_sources: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    desks_available: list[str]
    ipfs_enabled: bool
    onchain_enabled: bool


# ---------------------------------------------------------------------------
# Startup time tracking
# ---------------------------------------------------------------------------

_start_time = time.time()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for monitoring and load balancers."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=round(time.time() - _start_time, 1),
        desks_available=sorted(VALID_DESKS),
        ipfs_enabled=bool(os.getenv("ENABLE_IPFS", "false").lower() == "true"),
        onchain_enabled=bool(os.getenv("ENABLE_ONCHAIN", "false").lower() == "true"),
    )


@app.get("/api/v1/desks", response_model=list[DeskInfo])
async def list_desks():
    """List all available regional desks and their capabilities."""
    return [
        DeskInfo(
            name="US Equities",
            region="us",
            model="Groq Llama-3.3-70B",
            language="English",
            supported_tickers="US equities (AAPL, MSFT, NVDA, TSLA, etc.)",
            data_sources=["yfinance", "Financial Datasets API", "Stooq"],
        ),
        DeskInfo(
            name="China A-Shares",
            region="china",
            model="DeepSeek V4 Pro",
            language="Simplified Chinese (native reasoning)",
            supported_tickers="A-shares (600519.SH, 000858.SZ, etc.)",
            data_sources=["AKShare", "Tushare Pro", "Eastmoney"],
        ),
        DeskInfo(
            name="EU Equities",
            region="eu",
            model="Gemini 2.5 Flash",
            language="English (with local context)",
            supported_tickers="EU equities (MC.PA, SAP.DE, ASML.AS, etc.)",
            data_sources=["yfinance", "Stooq"],
        ),
        DeskInfo(
            name="Japan Equities",
            region="japan",
            model="Gemini 2.5 Flash",
            language="Japanese (native reasoning)",
            supported_tickers="JP equities (7203.T, 6758.T, 9984.T, etc.)",
            data_sources=["yfinance", "Stooq"],
        ),
        DeskInfo(
            name="Crypto",
            region="crypto",
            model="Groq Llama-3.3-70B",
            language="English",
            supported_tickers="Crypto (BTC, ETH, SOL, etc.)",
            data_sources=["CoinGecko", "DeFiLlama"],
        ),
    ]


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Generate a structured investment thesis for the given ticker and desk.

    This is the primary endpoint. It runs the regional agent's full analysis
    pipeline: data gathering, multi-sub-agent reasoning (fundamental, technical,
    sentiment), and thesis synthesis.

    Typical response time: 15-45 seconds depending on desk and data availability.
    """
    if request.desk not in VALID_DESKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid desk '{request.desk}'. Must be one of: {sorted(VALID_DESKS)}",
        )

    start = time.time()
    thesis_id = str(uuid.uuid4())

    try:
        agent = _get_agent(request.desk)
        thesis = await agent.analyze(request.ticker)

        # Optionally pin to IPFS
        ipfs_cid = None
        if os.getenv("ENABLE_IPFS", "false").lower() == "true":
            try:
                from src.persistence.multi_pinner import build_multi_pinner

                multi = build_multi_pinner()
                ipfs_cid, _ = await multi.pin(
                    thesis.model_dump(mode="json"),
                    name=f"{request.desk}-{request.ticker}-thesis",
                )
            except Exception as pin_err:
                logger.warning("IPFS pinning failed (non-blocking): %s", pin_err)

        processing_time = int((time.time() - start) * 1000)

        response = AnalyzeResponse(
            thesis_id=thesis_id,
            ticker=request.ticker,
            desk=request.desk,
            direction=thesis.direction.value,
            confidence=thesis.confidence_score,
            thesis_summary_en=thesis.thesis_summary_en,
            reasoning_blocks=[b.model_dump(mode="json") for b in thesis.reasoning_blocks],
            timestamp=thesis.timestamp.isoformat(),
            model_used=getattr(agent, "model_name", "unknown"),
            ipfs_cid=ipfs_cid,
            processing_time_ms=processing_time,
        )

        # Store for retrieval
        _thesis_store[thesis_id] = response.model_dump()

        return response

    except Exception as exc:
        logger.exception("Analysis failed for %s on %s desk", request.ticker, request.desk)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")


@app.get("/api/v1/thesis/{thesis_id}", response_model=AnalyzeResponse)
async def get_thesis(thesis_id: str):
    """Retrieve a previously generated thesis by its ID."""
    if thesis_id not in _thesis_store:
        raise HTTPException(status_code=404, detail="Thesis not found")
    return AnalyzeResponse(**_thesis_store[thesis_id])


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def _get_agent(desk: str):
    """Lazy-load the appropriate regional agent."""
    if desk == "us":
        from src.agents.us_agent import USAgent
        return USAgent()
    elif desk == "china":
        from src.agents.china_agent import ChinaAgent
        return ChinaAgent()
    elif desk == "eu":
        from src.agents.eu_agent import EUAgent
        return EUAgent()
    elif desk == "japan":
        from src.agents.japan_agent import JapanAgent
        return JapanAgent()
    elif desk == "crypto":
        from src.agents.crypto_agent import CryptoAgent
        return CryptoAgent()
    else:
        raise ValueError(f"Unknown desk: {desk}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run():
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "production") == "development",
    )


if __name__ == "__main__":
    run()
