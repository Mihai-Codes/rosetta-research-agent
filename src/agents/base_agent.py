"""Base class for all regional agents.

Each regional agent is an :class:`adal.Component` that:
1. Pulls locale-specific data via one or more MCP/HTTP clients.
2. Runs a chain of sub-agents (fundamental, technical, sentiment, macro,
   portfolio manager) — TradingAgents-style.
3. Synthesizes a structured :class:`InvestmentThesis`.

Subclasses override:
- :meth:`get_data_sources` — return the data the analyst sub-agents will see.
- :meth:`build_synthesis_prompt` — the portfolio-manager synthesis prompt.
- :attr:`region`, :attr:`working_language`, :attr:`default_model` — class attrs.

Why we use AdalFlow's Generator (not raw httpx → Anthropic):
- ``adal.Parameter`` wraps prompts so the AdalFlow Trainer can optimize them
  later via Textual Gradient Descent. Day-1 prompts are fine; Day-14 prompts
  will be auto-tuned.
- Provider-agnostic: swap Claude → DeepSeek for the China agent by changing
  ``model_client`` and ``model_kwargs``. No code changes downstream.
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from typing import Any

import json
import os
from pathlib import Path
from typing import TypeVar

# ---------------------------------------------------------------------------
# Learned guidelines — baked from text-grad feedback by training/bake_feedback.py
# Loaded once at import time; agents inject them as permanent synthesis rules.
# ---------------------------------------------------------------------------
_GUIDELINES_PATH = Path(__file__).parent.parent / "training" / "learned_guidelines.json"


def _load_guidelines() -> dict[str, list[str]]:
    try:
        if _GUIDELINES_PATH.exists():
            return json.loads(_GUIDELINES_PATH.read_text())
    except Exception as _e:
        pass
    return {}


LEARNED_GUIDELINES: dict[str, list[str]] = _load_guidelines()

import adalflow as adal
from pydantic import BaseModel

from reasoning.trace_schema import (
    AgentRole,
    AssetClass,
    Direction,
    InvestmentThesis,
    LangCode,
    ReasoningBlock,
    Region,
)

T = TypeVar("T", bound=BaseModel)


class PydanticJsonParser(adal.DataComponent):
    """Bridge: parses LLM raw-string JSON output into a Pydantic model.

    AdalFlow's JsonOutputParser requires adal.DataClass. Our domain models are
    Pydantic BaseModel for FastAPI/web3 interop. This thin wrapper sits at the
    Generator boundary and converts without touching the domain schemas.
    """

    def __init__(self, model_class: type[T]) -> None:
        super().__init__()
        self.model_class = model_class

    @staticmethod
    def _extract_json(raw: str) -> str:
        """Strip markdown code fences and return clean JSON string."""
        raw = raw.strip()
        if raw.startswith("```"):
            # handles ```json, ```JSON, ``` (no lang tag)
            raw = raw[3:]                      # strip opening ```
            if raw.startswith("json") or raw.startswith("JSON"):
                raw = raw[4:]
            # strip closing ```
            if raw.rstrip().endswith("```"):
                raw = raw.rstrip()[:-3]
        return raw.strip()

    def call(self, input: adal.GeneratorOutput, extra_fields: dict | None = None) -> T | None:  # noqa: A002
        """Parse GeneratorOutput → Pydantic model.

        Args:
            input: AdalFlow GeneratorOutput from the Generator call.
            extra_fields: fields to inject into the parsed dict before
                validation. Use this for fields the LLM shouldn't echo
                (e.g. agent_role, region).
        """
        raw = getattr(input, "raw_response", None) or getattr(input, "data", None) or str(input)
        if not raw:
            return None
        try:
            data = json.loads(self._extract_json(raw))
            if extra_fields:
                data.update(extra_fields)
            # reasoning_blocks is always re-injected by base_agent.analyze after parsing.
            # Force it to [] here to prevent validation errors when LLM returns strings/nulls.
            if self.model_class.__name__ == "InvestmentThesis":
                data["reasoning_blocks"] = []
            # Normalize common LLM field-name variations for PredictionMarketQuestion.
            if self.model_class.__name__ == "PredictionMarketQuestion":
                # LLMs sometimes return expiry_date instead of expiry
                if "expiry_date" in data and "expiry" not in data:
                    data["expiry"] = data.pop("expiry_date")
                # Ensure source fields exist (injected via extra_fields but sometimes LLM echoes them)
                data.pop("expiry_date", None)  # remove any stray alias
            return self.model_class.model_validate(data)
        except Exception as exc:
            logger.error("PydanticJsonParser failed: %s | raw: %.200s", exc, raw)
            return None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts — baseline (open-source). Optimized variants live in prompts/optimized/
# (gitignored) per the Warp playbook in AGENTS.md §5.
# ---------------------------------------------------------------------------

SUB_AGENT_TEMPLATE = """\
You are the {{ role }} for the {{ region }} desk at Rosetta Alpha.
Working language: {{ language }}.

Subject: {{ ticker }} ({{ asset_class }})

Available data:
{{ data_summary }}

Produce a JSON object with these fields:
- input_data_summary: short description of what you used
- analysis: your reasoning IN {{ language }}
- analysis_en: English translation (omit if language is 'en')
- conclusion: one-sentence bottom line
- confidence: 0.0–1.0
- language: '{{ language }}'

{{ output_format_str }}
"""

SYNTHESIS_TEMPLATE = """\
You are the portfolio manager for the {{ region }} desk.
You receive analyst reports below. Synthesize them into a single InvestmentThesis JSON object.
Be honest about disagreements — surface them in risk_factors.

Subject: {{ ticker }} ({{ asset_class }})
Working language: {{ language }}
{% if learned_guidelines %}
=== LEARNED GUIDELINES (apply permanently to every output) ===
{{ learned_guidelines }}
=== END GUIDELINES ===
{% endif %}
{% if prior_feedback %}
=== PRIOR OPTIMIZATION FEEDBACK (apply these improvements) ===
{{ prior_feedback }}
=== END FEEDBACK ===
{% endif %}
Analyst reports:
{{ analyst_reports }}

Respond with ONLY a valid JSON object (no markdown, no prose) with these exact fields:
- asset_class: string (e.g. "equity")
- ticker_or_asset: string
- thesis_summary_en: string (English summary)
- thesis_summary_native: string or null
- direction: "LONG" | "SHORT" | "NEUTRAL"
- confidence_score: float 0.0-1.0
- time_horizon_days: integer
- reasoning_blocks: [] (leave empty — filled by pipeline)
- data_sources_used: list of strings
- risk_factors: list of strings
- model_routing: {}
"""


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class RegionalAgent(adal.Component):
    """Abstract base. Subclass per region (US, CN, EU, JP, CRYPTO)."""

    # Class-level attributes — subclass overrides
    region: Region
    working_language: LangCode = "en"
    # Default to Groq (free tier, zero local compute) when running outside AdaL CLI.
    # Override in subclass for region-specific routing (e.g. DeepSeek for CN agent).
    default_model_client: type[adal.ModelClient] = adal.GroqAPIClient  # type: ignore[attr-defined]
    default_model_kwargs: dict[str, Any] = {"model": "llama-3.3-70b-versatile"}
    sub_agent_roles: tuple[AgentRole, ...] = (
        AgentRole.FUNDAMENTAL_ANALYST,
        AgentRole.TECHNICAL_ANALYST,
        AgentRole.SENTIMENT_ANALYST,
    )

    def __init__(
        self,
        *,
        model_client: adal.ModelClient | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        client = model_client or self.default_model_client()
        kwargs = model_kwargs or self.default_model_kwargs

        # Parsers stored separately so we can call them with extra_fields.
        self._block_parser = PydanticJsonParser(ReasoningBlock)
        self._thesis_parser = PydanticJsonParser(InvestmentThesis)

        # Sub-agent generator — one Generator instance, called per role.
        # No output_processors here — we parse manually via _block_parser /
        # _thesis_parser so we can inject extra_fields (e.g. agent_role).
        self.sub_agent = adal.Generator(
            model_client=client,
            model_kwargs=kwargs,
            template=SUB_AGENT_TEMPLATE,
        )

        # Synthesis generator — emits the full InvestmentThesis.
        self.synthesizer = adal.Generator(
            model_client=client,
            model_kwargs=kwargs,
            template=SYNTHESIS_TEMPLATE,
        )

    # -- subclass contract --------------------------------------------------

    @abstractmethod
    async def get_data_sources(self, ticker: str) -> dict[AgentRole, str]:
        """Return ``{role: data_summary_string}`` for each sub-agent role.

        Each sub-agent sees only its own slice — fundamental gets filings,
        technical gets price/indicator data, sentiment gets news, etc.
        """

    @property
    @abstractmethod
    def asset_class_for(self) -> AssetClass:
        """Default asset class for this region. Override in subclass."""

    # -- main entry point ---------------------------------------------------

    async def analyze(self, ticker: str, prior_feedback: str = "") -> InvestmentThesis:
        """Run the full pipeline → return a structured thesis.

        Args:
            ticker: Ticker symbol or asset name.
            prior_feedback: Aggregated text-grad suggestions from previous
                optimization rounds. Injected into the synthesis prompt so the
                portfolio-manager LLM can apply prior judge critiques.
        """
        logger.info("Analyzing %s on %s desk", ticker, self.region.value)

        data_by_role = await self.get_data_sources(ticker)

        # 1. Run each sub-agent in parallel could be a future optimization.
        #    Sequential for now — easier to debug and AdalFlow's Trainer wants
        #    deterministic ordering for textual-gradient computation.
        blocks: list[ReasoningBlock] = []
        for role in self.sub_agent_roles:
            data_summary = data_by_role.get(role, "(no data routed to this sub-agent)")
            sub_kwargs = {
                "role": role.value,
                "region": self.region.value,
                "language": self.working_language,
                "ticker": ticker,
                "asset_class": self.asset_class_for.value,
                "data_summary": data_summary,
            }
            # Retry on transient 503 errors (Gemini/provider load spikes).
            block_output = None
            for _sub_attempt in range(3):
                block_output = self.sub_agent(prompt_kwargs=sub_kwargs)
                if getattr(block_output, "error", None) and "503" in str(block_output.error):
                    _wait = 2 ** (_sub_attempt + 2)
                    logger.warning(
                        "Sub-agent 503 for %s/%s — retrying in %ds (attempt %d/3)",
                        role.value, ticker, _wait, _sub_attempt + 1,
                    )
                    await asyncio.sleep(_wait)
                    continue
                break
            # The LLM doesn't echo agent_role back — inject it post-parse.
            block = self._block_parser.call(block_output, extra_fields={"agent_role": role.value})
            if not isinstance(block, ReasoningBlock):
                raw_snippet = (getattr(block_output, 'raw_response', None) or '')[:300]
                raise RuntimeError(f"{role} sub-agent failed to produce a ReasoningBlock. raw: {raw_snippet}")
            blocks.append(block)

        # 2. Portfolio-manager synthesis.
        analyst_reports = "\n\n".join(
            f"[{b.agent_role.value}]\n{b.analysis}\nConclusion: {b.conclusion} (conf={b.confidence:.2f})"
            for b in blocks
        )
        # Synthesize with retry on transient 503 errors (Gemini free-tier spikes).
        # Inject baked learned guidelines for this desk as a pre-formatted string.
        # Using a string (not a list) avoids Jinja2 for-loop rendering issues in
        # regional templates that mix AdalFlow {{}} and Jinja2 {% %} syntax.
        _region_key = self.region.value.lower()
        _raw_guidelines = LEARNED_GUIDELINES.get(_region_key, [])
        _guidelines_str = "\n".join(f"- {g}" for g in _raw_guidelines) if _raw_guidelines else ""

        synthesis_kwargs = {
            "region": self.region.value,
            "language": self.working_language,
            "ticker": ticker,
            "asset_class": self.asset_class_for.value,
            "analyst_reports": analyst_reports,
            "prior_feedback": prior_feedback,
            "learned_guidelines": _guidelines_str,
        }
        import asyncio as _asyncio
        thesis_output = None
        for _attempt in range(3):
            thesis_output = self.synthesizer(prompt_kwargs=synthesis_kwargs)
            if getattr(thesis_output, "error", None) and "503" in str(thesis_output.error):
                wait = 2 ** (_attempt + 2)  # 4s, 8s, 16s
                logger.warning("Synthesizer 503 — retrying in %ds (attempt %d/3)", wait, _attempt + 1)
                await _asyncio.sleep(wait)
                continue
            break
        thesis = self._thesis_parser.call(
            thesis_output,
            extra_fields={"region": self.region.value, "ticker_or_asset": ticker},
        )
        if not isinstance(thesis, InvestmentThesis):
            raw_snippet = (getattr(thesis_output, 'raw_response', None) or '')[:300]
            raise RuntimeError(f"Synthesizer failed. raw: {raw_snippet}")

        # 3. Splice the sub-agent blocks back in (synthesizer may have summarized them).
        if not thesis.reasoning_blocks:
            thesis = thesis.model_copy(update={"reasoning_blocks": blocks})

        # 4. Fetch live price and inject entry_price_1e8 for on-chain market creation.
        # Normalize ticker for yfinance: Tushare uses .SH/.SZ suffixes but
        # Yahoo Finance expects .SS (Shanghai) — e.g. 600519.SH → 600519.SS.
        entry_price_1e8: int | None = None
        try:
            from data.yfinance_client import YFinanceClient as _YFC
            # Normalize ticker for yfinance:
            # - Tushare .SH → Yahoo .SS  (Shanghai A-shares)
            # - Bare crypto symbols (BTC, ETH) → BTC-USD, ETH-USD
            _yf_ticker = ticker.replace(".SH", ".SS")
            if _yf_ticker in ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE"):
                _yf_ticker = f"{_yf_ticker}-USD"
            price = await _YFC().get_current_price(_yf_ticker)
            if price and price > 0:
                entry_price_1e8 = int(price * 1e8)
                logger.debug("Live price for %s: %.4f → entry_price_1e8=%d", _yf_ticker, price, entry_price_1e8)
        except Exception as _price_exc:
            logger.debug("Price fetch skipped for %s: %s", ticker, _price_exc)

        # 5. Defensive: ensure region/language match what we configured.
        return thesis.model_copy(
            update={
                "region": self.region,
                "working_language": self.working_language,
                "asset_class": thesis.asset_class or self.asset_class_for,
                "entry_price_1e8": entry_price_1e8,
            }
        )

    # Convenience for callers that don't need a thesis, just a direction.
    async def quick_signal(self, ticker: str) -> Direction:
        thesis = await self.analyze(ticker)
        return thesis.direction
