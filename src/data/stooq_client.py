"""Stooq data client — free OHLCV fallback for EU and JP desks.

Uses pandas-datareader's Stooq reader (no API key required).
Covers 3,900+ TSE stocks and major European exchanges.

Ticker normalisation
--------------------
Stooq uses its own suffix convention, distinct from Yahoo Finance:
  Yahoo → Stooq
  7203.T  → 7203.jp   (Tokyo Stock Exchange)
  MC.PA   → mc.fr     (Euronext Paris)
  SAP.DE  → sap.de    (XETRA / Frankfurt)
  ENEL.MI → enel.it   (Borsa Italiana)
  AMS.AS  → ams.nl    (Euronext Amsterdam)
  SAN.MC  → san.es    (Madrid / BME)

Only get_daily is implemented — Stooq does not provide fundamentals or news.
Callers should soft-fail those and let the LLM reason with partial data.

Run standalone:
    uv run python -m data.stooq_client --ticker 7203.T
    uv run python -m data.stooq_client --ticker MC.PA
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Yahoo Finance suffix → Stooq suffix mapping
_SUFFIX_MAP: dict[str, str] = {
    # Japan
    ".T": ".jp",
    # France (Euronext Paris)
    ".PA": ".fr",
    # Germany (XETRA)
    ".DE": ".de",
    # Italy (Borsa Italiana)
    ".MI": ".it",
    # Netherlands (Euronext Amsterdam)
    ".AS": ".nl",
    # Spain (BME)
    ".MC": ".es",
    # Belgium (Euronext Brussels)
    ".BR": ".be",
    # Portugal (Euronext Lisbon)
    ".LS": ".pt",
    # Switzerland (SIX)
    ".SW": ".ch",
    # Sweden (Nasdaq Stockholm)
    ".ST": ".se",
    # Denmark (Nasdaq Copenhagen)
    ".CO": ".dk",
    # Finland (Nasdaq Helsinki)
    ".HE": ".fi",
    # Norway (Oslo Bors)
    ".OL": ".no",
    # Austria (Vienna)
    ".VI": ".at",
    # Poland (Warsaw)
    ".WA": ".pl",
    # Czech Republic (Prague)
    ".PR": ".cz",
    # Hungary (Budapest)
    ".BD": ".hu",
    # UK (London)
    ".L": ".uk",
}


def _to_stooq_ticker(yahoo_ticker: str) -> str:
    """Convert a Yahoo Finance ticker to Stooq format.

    Examples:
        7203.T  → 7203.jp
        MC.PA   → mc.fr
        SAP.DE  → sap.de
    """
    for yahoo_suffix, stooq_suffix in _SUFFIX_MAP.items():
        if yahoo_ticker.upper().endswith(yahoo_suffix.upper()):
            base = yahoo_ticker[: -len(yahoo_suffix)]
            return (base + stooq_suffix).lower()
    # Unknown suffix — pass through lowercase and hope for the best
    logger.debug("No Stooq suffix mapping for %s — passing through as-is", yahoo_ticker)
    return yahoo_ticker.lower()


class StooqClient:
    """Async OHLCV client backed by Stooq (via pandas-datareader).

    No API key required. Rate limit: generous (bulk downloads available).
    Only provides daily OHLCV — no fundamentals, no news.
    """

    async def get_daily(
        self, ticker: str, period: str = "10d"
    ) -> list[dict[str, Any]]:
        """Fetch recent OHLCV bars for *ticker* (Yahoo Finance format).

        Translates ticker to Stooq format internally.
        Returns same schema as YFinanceClient.get_daily for drop-in use.
        """
        stooq_ticker = _to_stooq_ticker(ticker)
        # Convert period string (e.g. "10d", "1mo") to a date range
        days = _period_to_days(period)
        end = datetime.today()
        start = end - timedelta(days=days + 7)  # extra buffer for weekends/holidays

        def _fetch() -> list[dict[str, Any]]:
            try:
                from pandas_datareader import data as pdr  # lazy import
            except ImportError as exc:
                raise ImportError(
                    "pandas-datareader is required for Stooq fallback. "
                    "Install with: pip install pandas-datareader"
                ) from exc

            df = pdr.DataReader(stooq_ticker, "stooq", start=start, end=end)
            if df.empty:
                return []
            # Stooq returns newest-first; reverse to chronological order
            df = df.sort_index()
            # Keep only the most recent `days` rows
            df = df.tail(days)
            df.index = df.index.strftime("%Y-%m-%d")
            df.columns = df.columns.str.lower()
            return df.reset_index().rename(columns={"index": "date"}).to_dict("records")

        try:
            result = await asyncio.to_thread(_fetch)
            logger.info("Stooq fallback get_daily(%s → %s): %d rows", ticker, stooq_ticker, len(result))
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stooq get_daily(%s → %s) failed: %s", ticker, stooq_ticker, exc)
            return []


def _period_to_days(period: str) -> int:
    """Convert yfinance period string to approximate calendar days.

    Supported: Nd (days), Nmo (months), Ny (years).
    Defaults to 14 days for unknown formats.
    """
    period = period.strip().lower()
    if period.endswith("d"):
        return int(period[:-1])
    if period.endswith("mo"):
        return int(period[:-2]) * 30
    if period.endswith("y"):
        return int(period[:-1]) * 365
    return 14


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


async def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ticker", default="7203.T", help="Yahoo-format ticker e.g. 7203.T, MC.PA"
    )
    args = parser.parse_args()

    client = StooqClient()
    daily = await client.get_daily(args.ticker)
    print(f"Stooq ticker: {_to_stooq_ticker(args.ticker)}")
    print(f"Rows: {len(daily)}")
    print(json.dumps(daily[:3], indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(_main())
