"""Async yfinance wrapper for JP and EU regional agents.

Uses asyncio.to_thread to run blocking yfinance calls without blocking the
event loop. Mirrors TushareClient's soft-fail convention.

Run standalone:
    uv run python -m data.yfinance_client --ticker 7203.T
    uv run python -m data.yfinance_client --ticker MC.PA
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class YFinanceClient:
    """Thin async wrapper around yfinance for international equities.

    No API key required — yfinance scrapes Yahoo Finance publicly.
    Rate-limit: ~2000 req/hour per IP. Soft-fails on network errors.
    """

    async def get_daily(self, ticker: str, period: str = "10d") -> list[dict[str, Any]]:
        """Fetch recent OHLCV bars for *ticker* (e.g. '7203.T', 'MC.PA').

        Returns a list of records: [{date, open, high, low, close, volume}, ...].
        """
        try:
            import yfinance as yf  # lazy import — only needed for JP/EU desks

            def _fetch() -> list[dict[str, Any]]:
                hist = yf.Ticker(ticker).history(period=period)
                if hist.empty:
                    return []
                hist.index = hist.index.strftime("%Y-%m-%d")
                return hist.reset_index().rename(columns=str.lower).to_dict("records")

            return await asyncio.to_thread(_fetch)
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance get_daily(%s) failed: %s", ticker, exc)
            return []

    async def get_info(self, ticker: str) -> dict[str, Any]:
        """Fetch company fundamentals: P/E, EPS, market cap, sector, etc."""
        try:
            import yfinance as yf

            def _fetch() -> dict[str, Any]:
                info = yf.Ticker(ticker).info
                # Keep only the most useful fields to avoid context bloat
                keys = [
                    "shortName", "sector", "industry", "country",
                    "marketCap", "trailingPE", "forwardPE", "trailingEps",
                    "revenueGrowth", "grossMargins", "operatingMargins",
                    "returnOnEquity", "debtToEquity", "currentRatio",
                    "52WeekChange", "beta", "dividendYield",
                    "longBusinessSummary",
                ]
                return {k: info.get(k) for k in keys if info.get(k) is not None}

            return await asyncio.to_thread(_fetch)
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance get_info(%s) failed: %s", ticker, exc)
            return {}

    async def get_current_price(self, ticker: str) -> float | None:
        """Return the latest close/current price for *ticker*.

        Uses fast_info for speed — falls back to history if unavailable.
        Returns None on failure so callers can soft-fail gracefully.
        """
        try:
            import yfinance as yf

            def _fetch() -> float | None:
                t = yf.Ticker(ticker)
                # fast_info is the cheapest yfinance call (single field)
                price = getattr(t.fast_info, "last_price", None)
                if price:
                    return float(price)
                # fallback: last close from 1-day history
                hist = t.history(period="1d")
                if not hist.empty:
                    return float(hist["Close"].iloc[-1])
                return None

            return await asyncio.to_thread(_fetch)
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance get_current_price(%s) failed: %s", ticker, exc)
            return None

    async def get_news(self, ticker: str) -> list[dict[str, Any]]:
        """Fetch recent news headlines for *ticker*."""
        try:
            import yfinance as yf

            def _fetch() -> list[dict[str, Any]]:
                items = yf.Ticker(ticker).news or []
                return [
                    {"title": n.get("content", {}).get("title", ""), "link": n.get("content", {}).get("canonicalUrl", {}).get("url", "")}
                    for n in items[:8]
                ]

            return await asyncio.to_thread(_fetch)
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance get_news(%s) failed: %s", ticker, exc)
            return []


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


async def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="7203.T", help="e.g. 7203.T (Toyota), MC.PA (LVMH)")
    args = parser.parse_args()

    client = YFinanceClient()
    daily, info, news = await asyncio.gather(
        client.get_daily(args.ticker),
        client.get_info(args.ticker),
        client.get_news(args.ticker),
    )

    print("Daily bars:", json.dumps(daily[:3], indent=2, default=str))
    print("Info:", json.dumps(info, indent=2, default=str))
    print("News:", json.dumps(news[:3], indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
