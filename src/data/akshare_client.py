"""AKShare client — free A-share data fallback for the China agent.

AKShare provides 20+ free Chinese market data sources (Eastmoney, Sina, etc.)
with no token required. Used when TUSHARE_TOKEN is not set.

Also handles Yahoo Finance symbol mapping: Tushare uses '600519.SH' but
yfinance needs '600519.SS' (Shanghai) or '000001.SZ' (Shenzhen).

Docs: https://akshare.akfamily.xyz/
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _tushare_to_yf(ts_code: str) -> str:
    """Convert Tushare ticker to yfinance symbol.

    Examples:
        600519.SH → 600519.SS  (Shanghai A-share)
        000001.SZ → 000001.SZ  (Shenzhen — yfinance accepts .SZ directly)
    """
    if ts_code.endswith(".SH"):
        return ts_code[:-3] + ".SS"
    return ts_code


def _tushare_to_akshare(ts_code: str) -> str:
    """Convert Tushare ticker to AKShare 6-digit code (no suffix)."""
    return ts_code.split(".")[0]


async def get_stock_daily(ts_code: str, limit: int = 10) -> list[dict[str, Any]]:
    """Fetch recent daily OHLCV via AKShare (Eastmoney backend).

    Falls back to an empty list on any error — consistent with AGENTS.md
    soft-fail convention.
    """
    code = _tushare_to_akshare(ts_code)
    try:
        import akshare as ak
        # Run blocking akshare call in executor to not block event loop
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                adjust="qfq",
            )
        )
        if df is None or df.empty:
            logger.warning("AKShare returned empty data for %s", ts_code)
            return []
        # Keep last `limit` rows and convert to records
        df = df.tail(limit)
        records = df.to_dict(orient="records")
        logger.info("AKShare: fetched %d daily bars for %s", len(records), ts_code)
        return records
    except Exception as exc:
        logger.warning("AKShare daily failed for %s: %s", ts_code, exc)
        return []


async def get_stock_info(ts_code: str) -> dict[str, Any]:
    """Fetch basic company info + key financial ratios via AKShare."""
    code = _tushare_to_akshare(ts_code)
    result: dict[str, Any] = {}
    try:
        import akshare as ak
        loop = asyncio.get_event_loop()

        # Individual stock fundamentals (PE, PB, market cap, etc.)
        df = await loop.run_in_executor(
            None,
            lambda: ak.stock_individual_info_em(symbol=code)
        )
        if df is not None and not df.empty:
            # stock_individual_info_em returns a 2-col DataFrame (item, value)
            result = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
            logger.info("AKShare: fetched company info for %s (%d fields)", ts_code, len(result))
    except Exception as exc:
        logger.warning("AKShare info failed for %s: %s", ts_code, exc)

    return result


async def get_stock_news(ts_code: str, limit: int = 8) -> list[dict[str, Any]]:
    """Fetch recent news for stock via AKShare (Eastmoney news feed)."""
    code = _tushare_to_akshare(ts_code)
    try:
        import akshare as ak
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: ak.stock_news_em(symbol=code)
        )
        if df is None or df.empty:
            return []
        records = df.head(limit).to_dict(orient="records")
        logger.info("AKShare: fetched %d news items for %s", len(records), ts_code)
        return records
    except Exception as exc:
        logger.warning("AKShare news failed for %s: %s", ts_code, exc)
        return []


async def get_cn_data_bundle(ts_code: str) -> dict[str, Any]:
    """Convenience: fetch daily bars + company info + news in parallel.

    Returns a dict with keys: 'daily', 'info', 'news'.
    Any failed sub-call returns empty container (soft-fail).
    """
    daily, info, news = await asyncio.gather(
        get_stock_daily(ts_code, limit=10),
        get_stock_info(ts_code),
        get_stock_news(ts_code, limit=6),
        return_exceptions=True,
    )
    return {
        "daily": daily if not isinstance(daily, Exception) else [],
        "info": info if not isinstance(info, Exception) else {},
        "news": news if not isinstance(news, Exception) else [],
    }
