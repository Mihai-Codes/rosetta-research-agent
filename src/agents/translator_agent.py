"""Translator Agent — converts an InvestmentThesis into a PredictionMarketQuestion.

This agent is region-agnostic: it accepts a thesis from any desk (US, CN, CRYPTO, EU)
and produces a Polymarket-shaped binary question that is:
- Time-bounded (expiry ≤ 90 days out)
- Oracle-resolvable (objective YES/NO criteria citing a data source)
- Language-normalised to English

Model routing (per AGENTS.md):
- Default: Groq / llama-3.3-70b-versatile (fast, free-tier)
- Override via model_client / model_kwargs constructor args.

Run standalone:
    uv run python -m agents.translator_agent --ticker AAPL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

import adalflow as adal

from agents.base_agent import PydanticJsonParser
from reasoning.trace_schema import Direction, InvestmentThesis, PredictionMarketQuestion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_TRANSLATOR_TEMPLATE = """\
You are a prediction-market question designer. Convert the investment thesis below into a single binary, oracle-resolvable prediction-market question.

Rules:
1. The question MUST be answerable YES or NO by a neutral third party using public data.
2. The question MUST be time-bounded — expiry within 30–90 days from today ({{today}}).
3. resolution_criteria MUST cite a specific, publicly accessible data source (e.g. "CoinGecko daily close", "Bloomberg consensus EPS", "PBOC official announcement").
4. category must be exactly one of: macro | earnings | policy | crypto | geopolitics
5. translation_confidence: 0.0–1.0 — how faithfully this question captures the thesis.

Investment thesis:
Ticker: {{ticker}} | Region: {{region}} | Direction: {{direction}} | Confidence: {{confidence_score}}
Current price: {{current_price}}
Summary: {{summary_en}}
Risks: {{risks}}

Output ONLY a valid JSON object with exactly these fields (no markdown, no extra text):
- question_text: string (the YES/NO question)
- resolution_criteria: string (objective, machine-checkable rule + data source)
- expiry: string (ISO datetime, e.g. "2026-08-01T00:00:00+00:00")
- category: string (one of: macro | earnings | policy | crypto | geopolitics)
- translation_confidence: float 0.0–1.0
- source_thesis_id: "{{thesis_id}}"
- source_language: "{{lang}}"
- translated_by_model: "{{model_name}}"\
"""


class TranslatorAgent(adal.Component):
    """Converts an InvestmentThesis → PredictionMarketQuestion via LLM."""

    def __init__(
        self,
        *,
        model_client: adal.ModelClient | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        client = model_client or adal.GroqAPIClient()  # type: ignore[attr-defined]
        kwargs = model_kwargs or {"model": "llama-3.3-70b-versatile", "temperature": 0.1, "max_tokens": 1024}

        schema_str = json.dumps(
            PredictionMarketQuestion.model_json_schema(), indent=2
        )
        template = _TRANSLATOR_TEMPLATE.replace("{{schema}}", schema_str)

        self._generator = adal.Generator(
            model_client=client,
            model_kwargs=kwargs,
            template=template,
        )
        self._parser = PydanticJsonParser(PredictionMarketQuestion)
        self._model_name = kwargs.get("model", "llama-3.3-70b-versatile")

    def translate(self, thesis: InvestmentThesis) -> PredictionMarketQuestion | None:
        """Synchronously translate a thesis into a prediction-market question."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        risks = "; ".join(thesis.risk_factors) if thesis.risk_factors else "none provided"

        # Inject live entry price so the LLM can form a specific, non-stale price target.
        if thesis.entry_price_1e8 is not None:
            current_price_str = f"{thesis.entry_price_1e8 / 1e8:,.2f} (at thesis creation)"
        else:
            current_price_str = "unknown (use directional question, not a price target)"

        output = self._generator(
            prompt_kwargs={
                "today": today,
                "ticker": thesis.ticker_or_asset,
                "region": thesis.region.value,
                "direction": thesis.direction.value,
                "confidence_score": f"{thesis.confidence_score:.2f}",
                "current_price": current_price_str,
                "summary_en": thesis.thesis_summary_en[:800],
                "risks": risks[:400],
                "thesis_id": thesis.thesis_id,
                "lang": thesis.working_language or "en",
                "model_name": self._model_name,
            }
        )

        question = self._parser.call(
            output,
            extra_fields={
                "source_thesis_id": thesis.thesis_id,
                "source_language": thesis.working_language or "en",
                "translated_by_model": self._model_name,
            },
        )

        if not isinstance(question, PredictionMarketQuestion):
            logger.warning("TranslatorAgent failed to parse PredictionMarketQuestion. raw: %s",
                           getattr(output, "raw_response", "")[:300])
            return None

        return question

    async def atranslate(self, thesis: InvestmentThesis) -> PredictionMarketQuestion | None:
        """Async wrapper — runs translate in executor to avoid blocking event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.translate, thesis)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


async def _main() -> None:
    from reasoning.trace_schema import AssetClass, Direction, Region

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Construct a dummy thesis for smoke-testing
    from datetime import datetime, timezone
    thesis = InvestmentThesis(
        ticker=args.ticker,
        region=Region.US,
        asset_class=AssetClass.EQUITY,
        direction=Direction.LONG,
        conviction=0.75,
        summary=f"{args.ticker} shows strong earnings momentum with AI tailwinds.",
        catalysts=["Q2 earnings beat expected", "AI capex cycle acceleration"],
        risk_factors=["Fed rate hike", "regulatory headwinds"],
        lang="en",
    )

    agent = TranslatorAgent()
    question = await agent.atranslate(thesis)
    if question:
        print(question.model_dump_json(indent=2))
    else:
        print("Translation failed — check logs.")


def run() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    run()
