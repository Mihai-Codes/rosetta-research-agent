"""Crypto/DeFi regional agent.

Data sources:
- CoinGecko (price, market cap, metadata for 15k+ coins)
- DefiLlama (protocol TVL, chain TVL, stablecoin flows)

Run standalone:
    uv run python -m agents.crypto_agent --asset bitcoin
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from dotenv import load_dotenv

load_dotenv(override=True)

from agents.base_agent import RegionalAgent
from data.mcp_client import BinanceClient, CoinGeckoClient, DefiLlamaClient
from reasoning.trace_schema import AgentRole, AssetClass, Region

logger = logging.getLogger(__name__)


class CryptoAgent(RegionalAgent):
    region = Region.CRYPTO
    working_language = "en"
    sub_agent_roles = (
        AgentRole.FUNDAMENTAL_ANALYST,  # on-chain metrics, tokenomics
        AgentRole.SENTIMENT_ANALYST,    # market sentiment, news
        AgentRole.MACRO_ANALYST,        # DeFi ecosystem, TVL trends
    )

    @property
    def asset_class_for(self) -> AssetClass:
        return AssetClass.CRYPTO

    def __init__(
        self,
        *,
        model_client: "adal.ModelClient | None" = None,
        model_kwargs: "dict | None" = None,
    ) -> None:
        import os
        from typing import Any
        import adalflow as adal  # noqa: F811

        if model_client is None:
            groq_key = os.environ.get("GROQ_API_KEY", "").strip()
            gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if groq_key:
                model_client = adal.GroqAPIClient()  # type: ignore[attr-defined]
                if model_kwargs is None:
                    model_kwargs = {"model": "llama-3.3-70b-versatile", "temperature": 0.2, "max_tokens": 2048}
            elif gemini_key:
                logger.warning("GROQ_API_KEY not set — falling back to Gemini for Crypto desk")
                from data.gemini_client import GeminiClient
                _gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
                model_client = GeminiClient(api_key=gemini_key)
                if model_kwargs is None:
                    model_kwargs = {"model": _gemini_model, "temperature": 0.2, "max_tokens": 2048}
            else:
                raise RuntimeError("Crypto desk requires GROQ_API_KEY or GEMINI_API_KEY.")
        super().__init__(model_client=model_client, model_kwargs=model_kwargs)

    # Map common ticker symbols → CoinGecko coin IDs
    _TICKER_TO_COINGECKO: dict[str, str] = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "binancecoin",
        "XRP": "ripple",
        "ADA": "cardano",
        "AVAX": "avalanche-2",
        "DOT": "polkadot",
        "MATIC": "matic-network",
        "LINK": "chainlink",
        "UNI": "uniswap",
        "DOGE": "dogecoin",
        "LTC": "litecoin",
        "ATOM": "cosmos",
        "NEAR": "near",
    }

    # Map common tickers → DefiLlama protocol slugs
    _TICKER_TO_DEFILLAMA: dict[str, str] = {
        "ETH": "ethereum",
        "BNB": "binance",
        "AVAX": "avalanche",
        "SOL": "solana",
        "MATIC": "polygon",
    }

    async def get_data_sources(self, ticker: str) -> dict[AgentRole, str]:
        """Pull CoinGecko + DefiLlama data concurrently.

        `ticker` can be a symbol (e.g. 'BTC', 'ETH') or a CoinGecko coin ID
        (e.g. 'bitcoin', 'ethereum'). Symbol normalization is applied automatically.
        """
        cg_id = self._TICKER_TO_COINGECKO.get(ticker.upper(), ticker.lower())
        dl_slug = self._TICKER_TO_DEFILLAMA.get(ticker.upper(), cg_id)

        async with CoinGeckoClient() as cg, DefiLlamaClient() as dl:
            coin_task = cg.get_coin(cg_id)
            # DefiLlama protocol slug often matches CoinGecko id — soft-fail if not
            protocol_task = dl.get_protocol(dl_slug)
            coin, protocol = await asyncio.gather(
                coin_task, protocol_task, return_exceptions=True
            )

        # ---- Binance fallback when CoinGecko is rate-limited or down ----
        binance_ticker: dict | None = None
        binance_klines: list | None = None
        if isinstance(coin, Exception):
            logger.warning("CoinGecko failed (%s) — fetching Binance fallback data", coin)
            try:
                async with BinanceClient() as bn:
                    binance_ticker, binance_klines = await asyncio.gather(
                        bn.get_ticker_24h(cg_id),
                        bn.get_klines(cg_id, interval="1d", limit=7),
                        return_exceptions=False,
                    )
            except Exception as bn_exc:
                logger.warning("Binance fallback also failed: %s", bn_exc)

        # ---- fundamental: tokenomics + market data from CoinGecko (or Binance) ----
        if isinstance(coin, Exception):
            if binance_ticker:
                fundamental_data = json.dumps({
                    "source": "Binance (CoinGecko unavailable)",
                    "symbol": binance_ticker.get("symbol"),
                    "lastPrice": binance_ticker.get("lastPrice"),
                    "priceChangePercent": binance_ticker.get("priceChangePercent"),
                    "quoteVolume": binance_ticker.get("quoteVolume"),
                    "highPrice": binance_ticker.get("highPrice"),
                    "lowPrice": binance_ticker.get("lowPrice"),
                    "recent_daily_ohlcv": binance_klines,
                }, indent=2)[:4000]
            else:
                fundamental_data = f"[CoinGecko ERROR: {coin}] [Binance fallback unavailable]"
        else:
            # Trim to relevant fields — full response can be 50KB+
            trimmed = {
                "id": coin.get("id"),
                "name": coin.get("name"),
                "symbol": coin.get("symbol"),
                "market_data": {
                    k: coin.get("market_data", {}).get(k)
                    for k in [
                        "current_price",
                        "market_cap",
                        "total_volume",
                        "circulating_supply",
                        "total_supply",
                        "price_change_percentage_24h",
                        "price_change_percentage_7d_in_currency",
                    ]
                },
                "categories": coin.get("categories", [])[:5],
                "description_snippet": (coin.get("description", {}).get("en", "") or "")[:500],
            }
            fundamental_data = json.dumps(trimmed, indent=2)[:4000]

        # ---- macro: DeFi protocol TVL + chain flows from DefiLlama ----
        if isinstance(protocol, Exception):
            macro_data = f"[DefiLlama protocol not found for '{ticker}': {protocol}]\n"
            macro_data += "Note: DefiLlama slug may differ from CoinGecko id. TVL data unavailable."
        else:
            trimmed_proto = {
                "name": protocol.get("name"),
                "tvl": protocol.get("tvl"),
                "chainTvls": protocol.get("chainTvls", {}),
                "category": protocol.get("category"),
                "description": (protocol.get("description") or "")[:300],
            }
            macro_data = json.dumps(trimmed_proto, indent=2)[:3000]

        # ---- sentiment: derive from CoinGecko community + market sentiment ----
        if isinstance(coin, Exception):
            if binance_ticker:
                sentiment_data = json.dumps({
                    "source": "Binance (CoinGecko unavailable)",
                    "priceChangePercent_24h": binance_ticker.get("priceChangePercent"),
                    "count": binance_ticker.get("count"),  # number of trades
                    "weightedAvgPrice": binance_ticker.get("weightedAvgPrice"),
                }, indent=2)
            else:
                sentiment_data = "[No sentiment data — CoinGecko and Binance both unavailable]"
        else:
            sentiment_data = json.dumps(
                {
                    "sentiment_votes_up_percentage": coin.get("sentiment_votes_up_percentage"),
                    "sentiment_votes_down_percentage": coin.get("sentiment_votes_down_percentage"),
                    "community_data": coin.get("community_data", {}),
                    "developer_data": {
                        k: coin.get("developer_data", {}).get(k)
                        for k in ["stars", "forks", "pull_request_contributors", "commit_count_4_weeks"]
                    },
                },
                indent=2,
            )[:2000]

        return {
            AgentRole.FUNDAMENTAL_ANALYST: fundamental_data,
            AgentRole.SENTIMENT_ANALYST: sentiment_data,
            AgentRole.MACRO_ANALYST: macro_data,
        }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the Crypto agent on a CoinGecko coin ID.")
    parser.add_argument("--asset", default="bitcoin")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    agent = CryptoAgent()
    thesis = await agent.analyze(args.asset)
    print(thesis.model_dump_json(indent=2))


def run() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    run()
