"""Pydantic schemas for reasoning traces and prediction-market questions.

These are the **canonical domain models**. Every regional agent produces an
:class:`InvestmentThesis`; the Translator Agent produces a
:class:`PredictionMarketQuestion`. Both are JSON-serialized with sorted keys
before hashing — see :mod:`reasoning.hasher` for the canonicalization rule.

Why Pydantic (not adal.DataClass)?
- We need FastAPI / JSON Schema / web3 ABI interop.
- AdalFlow's JsonOutputParser accepts Pydantic models at the LLM boundary.
- Keeping DataClass conversion at the boundary keeps domain code clean.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AssetClass(str, Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    FX = "fx"
    REAL_ESTATE = "real_estate"


class Region(str, Enum):
    US = "US"
    CN = "CN"
    EU = "EU"
    JP = "JP"
    CRYPTO = "CRYPTO"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class AgentRole(str, Enum):
    """Sub-agent roles inside a regional agent (TradingAgents pattern)."""

    FUNDAMENTAL_ANALYST = "fundamental_analyst"
    TECHNICAL_ANALYST = "technical_analyst"
    SENTIMENT_ANALYST = "sentiment_analyst"
    MACRO_ANALYST = "macro_analyst"
    RESEARCH_MANAGER = "research_manager"
    PORTFOLIO_MANAGER = "portfolio_manager"
    TRANSLATOR = "translator"


# ---------------------------------------------------------------------------
# Reasoning blocks
# ---------------------------------------------------------------------------

# ISO 639-1 two-letter language code (e.g. "en", "zh", "ja", "de", "fr").
LangCode = Annotated[str, StringConstraints(min_length=2, max_length=2, pattern=r"^[a-z]{2}$")]


class ReasoningBlock(BaseModel):
    """A single step in the agent's reasoning chain.

    Each sub-agent (fundamental, technical, etc.) emits one of these. The
    portfolio manager's block is the synthesis. Order in the parent thesis
    matters for hash reproducibility — see :mod:`reasoning.hasher`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_role: AgentRole
    input_data_summary: str = Field(
        description="Short description of what this sub-agent consumed (e.g. '10-Q for AAPL Q1-2026')."
    )
    thought_process: str | None = Field(
        default=None,
        description="Internal Chain-of-Thought / Thinking trace (R1-style)."
    )
    analysis: str = Field(description="The reasoning/analysis in the agent's working language.")
    analysis_en: str | None = Field(
        default=None,
        description="English translation if `language` != 'en'. Required for non-English agents.",
    )
    conclusion: str = Field(description="One-sentence bottom line.")
    confidence: float = Field(ge=0.0, le=1.0, description="Subjective confidence in this block.")
    language: LangCode = Field(default="en")


# ---------------------------------------------------------------------------
# Investment thesis — the primary output of every regional agent
# ---------------------------------------------------------------------------


class InvestmentThesis(BaseModel):
    """Structured output from any regional agent.

    Hashing target: the canonical JSON of this object (see
    :func:`reasoning.hasher.canonical_hash`) is what gets pinned to IPFS and
    recorded on Arc.
    """

    model_config = ConfigDict(extra="forbid")

    # Identity
    thesis_id: str = Field(default_factory=lambda: str(uuid4()))
    region: Region
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Subject
    asset_class: AssetClass
    ticker_or_asset: str = Field(description="e.g. 'AAPL', '600519.SH', 'BTC', 'EUR/USD'.")

    # Headline
    thesis_summary_en: str = Field(description="Always English. Translated if produced non-natively.")
    thesis_summary_native: str | None = Field(
        default=None, description="In the agent's working language if not English."
    )
    working_language: LangCode = Field(default="en")

    # Decision
    direction: Direction
    confidence_score: float = Field(ge=0.0, le=1.0)
    time_horizon_days: int = Field(gt=0, le=3650)

    # Reasoning chain
    reasoning_blocks: list[ReasoningBlock] = Field(default_factory=list)

    # Provenance
    data_sources_used: list[str] = Field(
        description="e.g. ['mcp:financial-datasets/get_earnings', 'tushare:daily']."
    )
    model_routing: dict[str, Any] = Field(
        default_factory=dict,
        description="Map of agent_role → LLM model used. e.g. {'fundamental_analyst': 'claude-sonnet-4.6'}.",
    )

    # Risk
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Known unknowns. Data-source failures MUST be surfaced here, not hidden.",
    )

    # Live price at creation time (fixed-point: price × 1e8, e.g. $79357.00 → 7935700000000)
    # Used by PredictionMarket.createMarket() as the entry price for resolution.
    # None = price unavailable at thesis creation time (oracle fills on resolve).
    entry_price_1e8: int | None = Field(
        default=None,
        description="Current asset price × 1e8 at thesis creation. Passed to on-chain market.",
    )

    # Versioning
    schema_version: str = Field(default="1.0.0")


# ---------------------------------------------------------------------------
# Translator output — Polymarket-shaped question
# ---------------------------------------------------------------------------


class PredictionMarketQuestion(BaseModel):
    """A binary, oracle-resolvable question generated from a thesis."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question_text: str = Field(
        description="Yes/no, time-bounded. e.g. 'Will PBOC cut MLF rate by ≥25bps before 2026-07-01?'"
    )
    resolution_criteria: str = Field(
        description="Objective, machine-checkable rule for YES/NO. Cite the data source."
    )
    expiry: datetime
    category: str = Field(description="'macro' | 'earnings' | 'policy' | 'crypto' | 'geopolitics'.")

    # Provenance back to the thesis that spawned this question
    source_thesis_id: str
    source_language: LangCode
    translated_by_model: str = Field(description="e.g. 'deepseek-v4-pro', 'gemini-3.1-pro'.")
    translation_confidence: float = Field(ge=0.0, le=1.0)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = Field(default="1.0.0")


# ---------------------------------------------------------------------------
# Portfolio view — cross-region aggregated snapshot
# ---------------------------------------------------------------------------


class PortfolioPosition(BaseModel):
    """One holding/signal derived from an InvestmentThesis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker_or_asset: str
    region: Region
    asset_class: AssetClass
    direction: Direction
    confidence_score: float = Field(ge=0.0, le=1.0)
    thesis_id: str
    thesis_summary_en: str
    signal: float = Field(
        ge=-1.0,
        le=1.0,
        description=(
            "Signed conviction signal: direction_numeric × confidence_score. "
            "LONG→+1, SHORT→-1, NEUTRAL→0, scaled by confidence."
        ),
    )


class PortfolioView(BaseModel):
    """Aggregated cross-region portfolio snapshot.

    Produced by :class:`portfolio.engine.PortfolioEngine`. Designed to be
    pinned to IPFS and eventually recorded in ``ReasoningRegistry.sol``.
    """

    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    positions: list[PortfolioPosition]

    # Aggregated signal
    net_signal: float = Field(
        ge=-1.0,
        le=1.0,
        description="Mean conviction-weighted signal across all positions.",
    )
    net_direction: Direction = Field(
        description="Discretised direction: |net_signal| ≥ 0.10 → LONG/SHORT, else NEUTRAL."
    )
    aggregate_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean confidence_score across all positions.",
    )

    # Surfaced risk
    top_risk_factors: list[str] = Field(
        description="Deduplicated, frequency-ranked risk factors from all theses."
    )

    schema_version: str = Field(default="1.0.0")


# ---------------------------------------------------------------------------
# Trace metadata — what gets recorded on Arc
# ---------------------------------------------------------------------------


class TraceMetadata(BaseModel):
    """The on-chain footprint of a reasoning trace.

    Mirrors the Solidity struct in `contracts/ReasoningRegistry.sol`.
    Keep field order stable — it's part of the ABI.
    """

    model_config = ConfigDict(extra="forbid")

    trace_hash: str = Field(pattern=r"^0x[0-9a-f]{64}$", description="0x-prefixed SHA-256 hex.")
    ipfs_cid: str = Field(description="IPFS CID v1, e.g. 'bafy...'.")
    region: Region
    asset_class: AssetClass
    timestamp: datetime
    submitter: str = Field(pattern=r"^0x[0-9a-fA-F]{40}$", description="Arc address (EVM).")
