"""US regional agent — reference implementation.

Data sources:
- Financial Datasets (fundamentals, SEC filings, news)
- Alpha Vantage (technicals — added in Sprint 1 Day 4)

This is the **reference** for every other regional agent. When in doubt about
how a regional agent should be structured, read this file.

Run standalone:
    uv run python -m agents.us_agent --ticker AAPL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)  # loads .env before any AdalFlow client init

import adalflow as adal

from agents.base_agent import RegionalAgent
from data.mcp_client import FinancialDatasetsClient
from reasoning.trace_schema import AgentRole, AssetClass, Region

logger = logging.getLogger(__name__)


class USAgent(RegionalAgent):
    region = Region.US
    working_language = "en"
    sub_agent_roles = (
        AgentRole.FUNDAMENTAL_ANALYST,
        AgentRole.SENTIMENT_ANALYST,
        # AgentRole.TECHNICAL_ANALYST  # ← enabled when Alpha Vantage client lands
    )

    @property
    def asset_class_for(self) -> AssetClass:
        return AssetClass.EQUITY

    def __init__(
        self,
        *,
        model_client: adal.ModelClient | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if model_client is None:
            groq_key = os.environ.get("GROQ_API_KEY", "").strip()
            gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if groq_key:
                model_client = adal.GroqAPIClient()  # type: ignore[attr-defined]
                if model_kwargs is None:
                    model_kwargs = {"model": "llama-3.3-70b-versatile", "temperature": 0.2, "max_tokens": 2048}
            elif gemini_key:
                logger.warning("GROQ_API_KEY not set — falling back to Gemini for US desk")
                from data.gemini_client import GeminiClient
                _gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
                model_client = GeminiClient(api_key=gemini_key)
                if model_kwargs is None:
                    model_kwargs = {"model": _gemini_model, "temperature": 0.2, "max_tokens": 2048}
            else:
                raise RuntimeError("US desk requires GROQ_API_KEY or GEMINI_API_KEY.")
        super().__init__(model_client=model_client, model_kwargs=model_kwargs)

    async def get_data_sources(self, ticker: str) -> dict[AgentRole, str]:
        async with FinancialDatasetsClient() as fd:
            # Pull in parallel — fundamentals + news. Snapshot follows.
            facts_task = fd.get_company_facts(ticker)
            news_task = fd.get_news(ticker, limit=8)
            facts, news = await asyncio.gather(facts_task, news_task, return_exceptions=True)

        # Soft-fail per AGENTS.md convention: surface failures in risk_factors
        # via the empty-string sentinel — the LLM will note "no data" naturally.
        fundamentals_summary = (
            json.dumps(facts, indent=2)[:4000]
            if not isinstance(facts, Exception)
            else f"[ERROR: {facts}]"
        )
        news_summary = (
            json.dumps(news, indent=2)[:4000]
            if not isinstance(news, Exception)
            else f"[ERROR: {news}]"
        )

        return {
            AgentRole.FUNDAMENTAL_ANALYST: fundamentals_summary,
            AgentRole.SENTIMENT_ANALYST: news_summary,
        }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the US agent on a ticker.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    agent = USAgent()
    thesis = await agent.analyze(args.ticker)
    print(thesis.model_dump_json(indent=2))


def run() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    run()
