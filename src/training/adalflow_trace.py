"""AdalFlow-compatible training dataset generator for Rosetta Alpha.

Every live pipeline run emits a RosettaTraceRecord to a JSONL dataset.
When the autonomous settler resolves a market, it calls update_outcome()
to label that record with the ground-truth result.

Over time this builds a dataset of:
  (ticker, region, data_context, reasoning_trace) → (direction, was_correct)

That dataset can be used for:
  1. Few-shot prompt injection (AdalFlow BootstrapFewShot)
  2. Fine-tuning a smaller model on high-confidence correct traces
  3. Evaluating agent drift over time

Self-improving loop:
  analyze() → log_thesis_run() → record in dataset.jsonl
  settler resolves → update_outcome() → label added
  next optimization round → load_dataset(labeled_only=True) → feed to trainer

Usage:
  # After each agent.analyze() call:
  from training.adalflow_trace import log_thesis_run
  await log_thesis_run(thesis, metadata, data_context_str)

  # After settler.settle() call:
  from training.adalflow_trace import update_outcome
  update_outcome(trace_hash, was_correct=True, exit_price_usd=192.34)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import adalflow as adal

from reasoning.trace_schema import Direction, InvestmentThesis, Region, TraceMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset path
# ---------------------------------------------------------------------------

_DATASET_DIR = Path(__file__).parent
DATASET_PATH = _DATASET_DIR / "rosetta_dataset.jsonl"


# ---------------------------------------------------------------------------
# RosettaTraceRecord — one row in the dataset
# Designed as an AdalFlow DataClass for compatibility with BootstrapFewShot.
# ---------------------------------------------------------------------------

@adal.dataclass(eq=True, unsafe_hash=True)
class RosettaTraceRecord(adal.DataClass):
    """A single training example: reasoning trace → market outcome.

    Fields prefixed with `input_` are features; `output_` are labels.
    AdalFlow's few-shot optimizer reads __input_fields__ / __output_fields__.
    """

    # ── Identity ────────────────────────────────────────────────────────────
    trace_hash:    str = field(default="", metadata={"desc": "SHA-256 of the thesis JSON"})
    ipfs_cid:      str = field(default="", metadata={"desc": "IPFS CID of the full trace"})
    arc_tx:        str = field(default="", metadata={"desc": "Arc on-chain record TX hash"})
    market_tx:     str = field(default="", metadata={"desc": "PredictionMarket create TX hash"})
    run_timestamp: str = field(default="", metadata={"desc": "ISO-8601 UTC timestamp of the run"})

    # ── Input features ──────────────────────────────────────────────────────
    ticker:        str   = field(default="", metadata={"desc": "Asset ticker"})
    region:        str   = field(default="", metadata={"desc": "Region enum value"})
    language:      str   = field(default="en", metadata={"desc": "Native analysis language"})
    entry_price_usd: float = field(default=0.0, metadata={"desc": "Asset price at thesis creation"})
    data_context:  str   = field(default="", metadata={"desc": "Raw data passed to agent (truncated to 2k chars)"})

    # ── Reasoning trace (the product) ───────────────────────────────────────
    thought_process:    str = field(default="", metadata={"desc": "R1-style chain-of-thought reasoning"})
    thesis_summary_en:  str = field(default="", metadata={"desc": "English thesis summary"})
    thesis_summary_native: str = field(default="", metadata={"desc": "Native-language thesis summary"})
    key_risks:          str = field(default="", metadata={"desc": "Comma-separated key risks"})

    # ── Output labels ───────────────────────────────────────────────────────
    direction:       str   = field(default="", metadata={"desc": "LONG / SHORT / NEUTRAL"})
    confidence:      float = field(default=0.0, metadata={"desc": "Confidence score 0–1"})
    horizon_days:    int   = field(default=30,  metadata={"desc": "Time horizon in days"})

    # ── Market outcome (filled by settler) ──────────────────────────────────
    was_correct:     bool | None  = field(default=None, metadata={"desc": "True if market resolved correctly"})
    exit_price_usd:  float | None = field(default=None, metadata={"desc": "Asset price at settlement"})
    price_change_pct: float | None = field(default=None, metadata={"desc": "% price change entry→exit"})
    settled_at:      str | None   = field(default=None, metadata={"desc": "ISO-8601 UTC settlement timestamp"})

    # AdalFlow few-shot interface
    __input_fields__  = ["ticker", "region", "data_context", "thought_process", "thesis_summary_en"]
    __output_fields__ = ["direction", "confidence", "was_correct"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated {len(text) - max_chars} chars]"


def _read_all_records() -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        return []
    records = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _write_all_records(records: list[dict[str, Any]]) -> None:
    with DATASET_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_thesis_run(
    thesis: InvestmentThesis,
    metadata: TraceMetadata,
    data_context: str = "",
    market_tx: str = "",
) -> RosettaTraceRecord:
    """Append a new training record after a live analyze() + record_trace() call.

    Call this immediately after arc_recorder.record_trace() returns, so you
    have both the IPFS CID and Arc TX hash to persist.

    Args:
        thesis:       The InvestmentThesis returned by agent.analyze().
        metadata:     The TraceMetadata returned by the trace pipeline
                      (contains trace_hash, ipfs_cid, arc_tx).
        data_context: Raw string of data passed to the agent (prices, filings).
                      Will be truncated to 2k chars to keep dataset lean.
        market_tx:    PredictionMarket.createMarket() TX hash (optional).

    Returns:
        The RosettaTraceRecord that was appended.
    """
    # Extract R1 thought_process from reasoning blocks if present
    thought = ""
    if thesis.reasoning:
        thoughts = [b.thought_process for b in thesis.reasoning if b.thought_process]
        if thoughts:
            thought = "\n\n---\n\n".join(thoughts)

    # Build risks string
    risks = ", ".join(thesis.key_risks) if thesis.key_risks else ""

    record = RosettaTraceRecord(
        trace_hash    = metadata.trace_hash,
        ipfs_cid      = metadata.ipfs_cid or "",
        arc_tx        = getattr(metadata, "arc_tx", "") or "",
        market_tx     = market_tx or "",
        run_timestamp = _now_iso(),
        ticker        = thesis.ticker_or_asset,
        region        = metadata.region.value,
        language      = thesis.language.value if thesis.language else "en",
        entry_price_usd = (thesis.entry_price_1e8 / 1e8) if thesis.entry_price_1e8 else 0.0,
        data_context  = _truncate(data_context),
        thought_process    = _truncate(thought, 4000),
        thesis_summary_en  = thesis.thesis_summary_en or "",
        thesis_summary_native = thesis.thesis_summary_native or "",
        key_risks     = risks,
        direction     = thesis.direction.value,
        confidence    = thesis.confidence_score,
        horizon_days  = int(thesis.time_horizon_days),
        # outcome fields — filled later by settler
        was_correct   = None,
        exit_price_usd = None,
        price_change_pct = None,
        settled_at    = None,
    )

    # Append to JSONL
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    logger.info(
        "Dataset record logged: %s %s %s conf=%.0f%% → %s",
        thesis.ticker_or_asset,
        metadata.region.value,
        thesis.direction.value,
        thesis.confidence_score * 100,
        DATASET_PATH,
    )
    return record


def update_outcome(
    trace_hash: str,
    was_correct: bool,
    exit_price_usd: float | None = None,
) -> bool:
    """Label a previously logged record with its market outcome.

    Called by reasoning/settler.py after settle() confirms resolution.

    Args:
        trace_hash:     The SHA-256 trace hash (matches record.trace_hash).
        was_correct:    True if the market resolved in the agent's favour.
        exit_price_usd: Exit price at settlement (optional, for analytics).

    Returns:
        True if the record was found and updated, False otherwise.
    """
    records = _read_all_records()
    updated = False

    for rec in records:
        if rec.get("trace_hash") == trace_hash:
            entry = rec.get("entry_price_usd", 0.0) or 0.0
            pct_change: float | None = None
            if exit_price_usd and entry > 0:
                pct_change = (exit_price_usd - entry) / entry * 100

            rec["was_correct"]      = was_correct
            rec["exit_price_usd"]   = exit_price_usd
            rec["price_change_pct"] = pct_change
            rec["settled_at"]       = _now_iso()
            updated = True
            break

    if updated:
        _write_all_records(records)
        logger.info(
            "Dataset outcome updated: %s → correct=%s exit=$%s",
            trace_hash[:16],
            was_correct,
            f"{exit_price_usd:.2f}" if exit_price_usd else "N/A",
        )
    else:
        logger.warning("update_outcome: trace_hash not found in dataset: %s", trace_hash[:16])

    return updated


def load_dataset(
    labeled_only: bool = False,
    region: str | None = None,
    min_confidence: float = 0.0,
) -> list[RosettaTraceRecord]:
    """Load dataset records, optionally filtered.

    Args:
        labeled_only:    If True, only return records with a known outcome.
        region:          Filter by region value (e.g. "US", "CN").
        min_confidence:  Only return records above this confidence threshold.

    Returns:
        List of RosettaTraceRecord objects.
    """
    raw = _read_all_records()
    records: list[RosettaTraceRecord] = []

    for r in raw:
        if labeled_only and r.get("was_correct") is None:
            continue
        if region and r.get("region") != region:
            continue
        if r.get("confidence", 0.0) < min_confidence:
            continue
        try:
            records.append(RosettaTraceRecord(**r))
        except Exception as exc:
            logger.debug("Skipping malformed record: %s", exc)

    return records


def dataset_stats() -> dict[str, Any]:
    """Quick summary stats for logging / dashboard display."""
    records = _read_all_records()
    labeled = [r for r in records if r.get("was_correct") is not None]
    correct = [r for r in labeled if r.get("was_correct") is True]

    by_region: dict[str, int] = {}
    for r in records:
        rgn = r.get("region", "UNKNOWN")
        by_region[rgn] = by_region.get(rgn, 0) + 1

    accuracy = len(correct) / len(labeled) if labeled else None

    return {
        "total":        len(records),
        "labeled":      len(labeled),
        "unlabeled":    len(records) - len(labeled),
        "correct":      len(correct),
        "accuracy":     round(accuracy * 100, 1) if accuracy is not None else None,
        "by_region":    by_region,
        "dataset_path": str(DATASET_PATH),
    }


# ---------------------------------------------------------------------------
# CLI — quick inspection
# ---------------------------------------------------------------------------

def main() -> None:
    """python -m training.adalflow_trace [--stats] [--labeled]"""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Inspect the Rosetta training dataset")
    parser.add_argument("--stats", action="store_true", help="Print summary stats")
    parser.add_argument("--labeled", action="store_true", help="Print labeled records only")
    parser.add_argument("--region", default=None, help="Filter by region (US, CN, EU, JP, CRYPTO)")
    args = parser.parse_args()

    if args.stats:
        stats = dataset_stats()
        print(json.dumps(stats, indent=2))
        return

    records = load_dataset(labeled_only=args.labeled, region=args.region)
    print(f"Loaded {len(records)} record(s){' (labeled only)' if args.labeled else ''}:")
    for rec in records:
        outcome = "✅" if rec.was_correct else ("❌" if rec.was_correct is False else "⏳")
        print(
            f"  {outcome} {rec.ticker:12s} {rec.region:6s} {rec.direction:8s} "
            f"conf={rec.confidence*100:.0f}%  {rec.run_timestamp[:10]}"
        )


if __name__ == "__main__":
    main()
