"""EU regional agent — pan-European equity analysis (English + native language).

Data sources:
- yfinance (daily OHLCV, company fundamentals, news headlines)

Model routing (per AGENTS.md):
- Gemini 2.5 Pro — multilingual EU analysis (FR/DE/IT/ES/NL).
  Prompts are in English; the agent summarises native-language filings via
  Gemini's multimodal and multilingual capabilities.
  Override with EU_AGENT_MODEL env var.

Run standalone:
    GEMINI_API_KEY=xxx uv run python -m agents.eu_agent --ticker MC.PA
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

import adalflow as adal

from agents.base_agent import PydanticJsonParser, RegionalAgent
from data.stooq_client import StooqClient
from data.yfinance_client import YFinanceClient
from reasoning.trace_schema import (
    AgentRole,
    AssetClass,
    InvestmentThesis,
    ReasoningBlock,
    Region,
)

logger = logging.getLogger(__name__)

# Override with EU_AGENT_MODEL (e.g. "gemini-2.5-flash" for cheaper iteration).
_GEMINI_MODEL = os.environ.get("EU_AGENT_MODEL", "gemini-2.5-pro")

# ---------------------------------------------------------------------------
# Prompt templates — English primary, native secondary
# ---------------------------------------------------------------------------

_EU_SUB_AGENT_TEMPLATE = """\
<SYS>
You are an institutional equity analyst specialising in European markets (EU/EEA).
Your role is: {{role}}.
Analyse the data below focusing on earnings quality, ECB policy sensitivity,
FX exposure (EUR/local), regulatory risk, and ESG positioning.

Output MUST be strictly JSON only (no extra text):
{"agent_role": "{{role}}", "input_data_summary": "<50 chars: data consumed>", "analysis": "<150 chars max analysis>", "analysis_en": "<analysis in English>", "conclusion": "<one sentence>", "confidence": <0.0-1.0>, "language": "en"}
</SYS>

[Ticker] {{ticker}} | [Market] {{region}} | [Asset Class] {{asset_class}}

{{data_summary}}\
"""

_EU_SYNTHESIS_TEMPLATE = """\
<SYS>
You are a senior European portfolio manager. Synthesise the analyst reports below
into a final investment recommendation following this exact JSON schema:
{{schema}}

Rules:
- direction: LONG, SHORT, or NEUTRAL (uppercase)
- confidence_score: 0.0–1.0
- region: EU
- asset_class: equity
- working_language: en
- ticker_or_asset: the ticker symbol
Output JSON only — no markdown, no prose.
</SYS>

[Ticker] {{ticker}} | [Market] {{region}} | [Asset Class] {{asset_class}}
{% if learned_guidelines %}
=== LEARNED GUIDELINES (apply permanently to every output) ===
{{learned_guidelines}}
=== END GUIDELINES ===
{% endif %}
{% if prior_feedback %}
=== PRIOR OPTIMIZATION FEEDBACK (apply these improvements) ===
{{prior_feedback}}
=== END FEEDBACK ===
{% endif %}
Analyst reports:
{{analyst_reports}}\
"""


class EUAgent(RegionalAgent):
    """Pan-European equity analyst — Gemini-powered multilingual analysis."""

    region = Region.EU
    working_language = "en"
    sub_agent_roles = (
        AgentRole.FUNDAMENTAL_ANALYST,
        AgentRole.MACRO_ANALYST,
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
            gemini_key = os.environ.get("GEMINI_API_KEY", "")
            if gemini_key:
                from data.gemini_client import GeminiClient
                model_client = GeminiClient(api_key=gemini_key)
                if model_kwargs is None:
                    model_kwargs = {"model": _GEMINI_MODEL, "temperature": 0.2, "max_tokens": 2048}
            else:
                logger.warning("GEMINI_API_KEY not set — falling back to Groq for EU desk")
                model_client = adal.GroqAPIClient()  # type: ignore[attr-defined]
                if model_kwargs is None:
                    model_kwargs = {"model": "llama-3.3-70b-versatile", "temperature": 0.2, "max_tokens": 2048}

        if model_kwargs is None:
            model_kwargs = {"model": _GEMINI_MODEL, "temperature": 0.2, "max_tokens": 2048}

        super().__init__(model_client=model_client, model_kwargs=model_kwargs)

        schema_str = json.dumps(
            InvestmentThesis.model_json_schema(), ensure_ascii=False, indent=2
        )
        eu_synthesis = _EU_SYNTHESIS_TEMPLATE.replace("{{schema}}", schema_str)

        self.sub_agent = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=_EU_SUB_AGENT_TEMPLATE,
        )
        self.synthesizer = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=eu_synthesis,
        )

        self._block_parser = PydanticJsonParser(ReasoningBlock)
        self._thesis_parser = PydanticJsonParser(InvestmentThesis)

    # ------------------------------------------------------------------
    # Data sources — yfinance
    # ------------------------------------------------------------------

    async def get_data_sources(self, ticker: str) -> dict[AgentRole, str]:
        """Pull daily bars + fundamentals + news from yfinance (Stooq fallback for price data)."""
        yf_client = YFinanceClient()
        daily, info, news = await asyncio.gather(
            yf_client.get_daily(ticker, period="10d"),
            yf_client.get_info(ticker),
            yf_client.get_news(ticker),
            return_exceptions=True,
        )

        # Price data: fall back to Stooq if yfinance fails or returns nothing
        if isinstance(daily, Exception) or not daily:
            logger.warning("yfinance get_daily failed for %s — trying Stooq fallback", ticker)
            daily = await StooqClient().get_daily(ticker, period="10d")

        if isinstance(daily, Exception) or not daily:
            daily_summary = "[Price Data] Unavailable"
        else:
            rows = json.dumps(daily[:10], ensure_ascii=False)[:2000]
            daily_summary = f"[Price History — last 10 trading days]\n{rows}"

        if isinstance(info, Exception) or not info:
            funda_summary = "[Fundamentals] Unavailable"
        else:
            rows = json.dumps(info, ensure_ascii=False)[:1500]
            funda_summary = f"[Company Fundamentals]\n{rows}"

        if isinstance(news, Exception) or not news:
            news_summary = "[News] Unavailable"
        else:
            rows = json.dumps(news[:5], ensure_ascii=False)[:800]
            news_summary = f"[Recent News]\n{rows}"

        combined = f"{daily_summary}\n\n{funda_summary}\n\n{news_summary}"

        return {
            AgentRole.FUNDAMENTAL_ANALYST: combined,
            AgentRole.MACRO_ANALYST: combined,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="MC.PA", help="EU ticker e.g. MC.PA (LVMH), SAP.DE")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    agent = EUAgent()
    thesis = await agent.analyze(args.ticker)
    print(thesis.model_dump_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(_main())
