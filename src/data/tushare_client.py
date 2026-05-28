"""Tushare Pro client — A-share market data for the China agent.

Tushare Pro provides:
- Daily stock quotes (daily)
- Company fundamentals (fina_indicator)
- News feed (news)

Docs: https://tushare.pro/document/2

Run standalone:
    TUSHARE_TOKEN=xxx uv run python -m data.tushare_client --ticker 600519.SH
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "http://api.tushare.pro"


class TushareClient:
    """Async HTTP wrapper around the Tushare Pro REST API.

    All methods soft-fail: they return an empty dict / list on network or
    auth errors, consistent with the AGENTS.md convention for data clients.
    """

    def __init__(self, token: str | None = None) -> None:
        self._token = token or os.environ.get("TUSHARE_TOKEN", "")
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "TushareClient":
        self._http = httpx.AsyncClient(timeout=15)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._http:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post(self, api_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """POST to Tushare API and return the ``data`` section."""
        if not self._token:
            logger.warning("TUSHARE_TOKEN not set — returning empty result")
            return {}
        if self._http is None:
            raise RuntimeError("Use TushareClient as async context manager")

        payload = {
            "api_name": api_name,
            "token": self._token,
            "params": params,
            "fields": "",
        }
        try:
            resp = await self._http.post(_BASE_URL, json=payload)
            resp.raise_for_status()
            body = resp.json()
            if body.get("code") != 0:
                logger.warning("Tushare API error [%s]: %s", api_name, body.get("msg"))
                return {}
            raw = body.get("data", {})
            # Tushare returns {fields: [...], items: [[...]]}
            fields = raw.get("fields", [])
            items = raw.get("items", [])
            if not fields:
                return raw
            return {"fields": fields, "items": items}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tushare %s failed: %s", api_name, exc)
            return {}

    @staticmethod
    def _to_records(raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert Tushare field/items structure to list of dicts."""
        fields = raw.get("fields", [])
        items = raw.get("items", [])
        if not fields:
            return []
        return [dict(zip(fields, row)) for row in items]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_daily(self, ts_code: str, limit: int = 10) -> list[dict[str, Any]]:
        """Fetch recent daily OHLCV bars for *ts_code* (e.g. ``600519.SH``)."""
        raw = await self._post("daily", {"ts_code": ts_code, "limit": limit})
        return self._to_records(raw)

    async def get_fina_indicator(self, ts_code: str) -> list[dict[str, Any]]:
        """Fetch key financial indicators (ROE, EPS, gross margin, etc.)."""
        raw = await self._post(
            "fina_indicator",
            {"ts_code": ts_code, "limit": 4},  # last 4 reporting periods
        )
        return self._to_records(raw)

    async def get_news(self, keyword: str, limit: int = 8) -> list[dict[str, Any]]:
        """Fetch financial news mentioning *keyword* from Tushare news feed."""
        raw = await self._post("news", {"src": "sina", "limit": limit})
        records = self._to_records(raw)
        # Filter by keyword client-side (Tushare news endpoint lacks full-text filter)
        kw = keyword.lower()
        filtered = [r for r in records if kw in str(r).lower()]
        return filtered or records[:limit]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


async def _main() -> None:
    import argparse
    import json

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="600519.SH")
    args = parser.parse_args()

    async with TushareClient() as ts:
        daily, fina = await asyncio.gather(
            ts.get_daily(args.ticker),
            ts.get_fina_indicator(args.ticker),
        )

    print("Daily bars:", json.dumps(daily[:3], ensure_ascii=False, indent=2))
    print("Fina indicators:", json.dumps(fina[:2], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
