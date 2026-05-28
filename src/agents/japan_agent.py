"""Japan regional agent — TSE/JPX equity analysis in Japanese.

Data sources:
- yfinance (daily OHLCV, company fundamentals, news headlines)

Model routing (per AGENTS.md):
- Gemini 2.5 Pro — multilingual Japanese reasoning + multimodal chart support.
  Prompts are in Japanese so Gemini uses its strongest CJK reasoning path.
  Override with JP_AGENT_MODEL env var.

Run standalone:
    GEMINI_API_KEY=xxx uv run python -m agents.japan_agent --ticker 7203.T
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

# Gemini via AdalFlow's GoogleGenAIClient.
# Override with JP_AGENT_MODEL (e.g. "gemini-2.5-flash" for cheaper runs).
_GEMINI_MODEL = os.environ.get("JP_AGENT_MODEL", "gemini-2.5-pro")

# ---------------------------------------------------------------------------
# Japanese-language prompt templates
# ---------------------------------------------------------------------------

_JP_SUB_AGENT_TEMPLATE = """\
<SYS>
あなたは東証（TSE）専門の機関投資家アナリストです。役割：{{role}}。
以下の市場データを基に、収益性、バリュエーション、テクニカル、マクロ政策の観点から深く分析してください。
深い思考プロセス（R1スタイルのChain-of-Thought）を示した上で、最終的な分析と英語の翻訳を提供してください。

出力は必ず厳密なJSONのみ（余分なテキスト不可）：
{"agent_role": "{{role}}", "input_data_summary": "<50文字以内のデータ概要>", "thought_process": "<論理的推論、賛否両論の検討、データによる裏付けなど、少なくとも200文字の詳細な思考プロセス>", "analysis": "<最終的な分析結果、100〜200文字>", "analysis_en": "<English translation of the final analysis>", "conclusion": "<one sentence>", "confidence": <0.0-1.0>, "language": "ja"}
</SYS>

【銘柄】{{ticker}} | 【市場】{{region}} | 【資産クラス】{{asset_class}}

{{data_summary}}\
"""

_JP_SYNTHESIS_TEMPLATE = """\
<SYS>
あなたはシニアポートフォリオマネージャーです。複数のアナリストレポートを統合し、最終的な投資判断を下してください。

出力は必ず以下のスキーマに準拠した厳密なJSONのみ（余分なテキスト不可）：
{{schema}}

重要事項：
- direction: LONG、SHORT、またはNEUTRAL（大文字英語）
- confidence_score: 0.0〜1.0
- region: JP
- asset_class: equity
- working_language: ja
- ticker_or_asset: 銘柄コードを記入
JSONのみ出力すること。
</SYS>

【銘柄】{{ticker}} | 【市場】{{region}} | 【資産クラス】{{asset_class}}
{% if learned_guidelines %}
=== 学習済みガイドライン（常にすべての出力に適用） ===
{{learned_guidelines}}
=== ガイドライン終了 ===
{% endif %}
{% if prior_feedback %}
=== 前回最適化フィードバック（これを反映してください） ===
{{prior_feedback}}
=== フィードバック終了 ===
{% endif %}
アナリストレポート集約：
{{analyst_reports}}\
"""


class JapanAgent(RegionalAgent):
    """TSE analyst — runs all reasoning in Japanese via Gemini."""

    region = Region.JP
    working_language = "ja"
    sub_agent_roles = (
        AgentRole.FUNDAMENTAL_ANALYST,
        AgentRole.TECHNICAL_ANALYST,
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
                logger.warning("GEMINI_API_KEY not set — falling back to Groq for JP desk")
                model_client = adal.GroqAPIClient()  # type: ignore[attr-defined]
                if model_kwargs is None:
                    model_kwargs = {"model": "llama-3.3-70b-versatile", "temperature": 0.2, "max_tokens": 2048}

        if model_kwargs is None:
            model_kwargs = {"model": _GEMINI_MODEL, "temperature": 0.2, "max_tokens": 2048}

        super().__init__(model_client=model_client, model_kwargs=model_kwargs)

        # Use a compact field list instead of the full JSON schema to stay within
        # Gemini's 2048-token prompt limit (full schema = ~2100 tokens).
        schema_str = """{
  "region": "JP",
  "asset_class": "equity",
  "ticker_or_asset": "<銘柄コード>",
  "thesis_summary_en": "<English summary>",
  "thesis_summary_native": "<日本語サマリー>",
  "working_language": "ja",
  "direction": "LONG",
  "confidence_score": 0.75,
  "time_horizon_days": 180,
  "reasoning_blocks": [
    {
      "agent_role": "fundamental_analyst",
      "input_data_summary": "<データ概要>",
      "thought_process": "<詳細な思考プロセス>",
      "analysis": "<日本語分析>",
      "analysis_en": "<English translation>",
      "conclusion": "<one sentence conclusion>",
      "confidence": 0.8,
      "language": "ja"
    }
  ],
  "data_sources_used": ["yfinance:daily", "yfinance:info"],
  "risk_factors": ["<リスク要因>"],
  "model_routing": {},
  "schema_version": "1.0.0"
}"""
        jp_synthesis = _JP_SYNTHESIS_TEMPLATE.replace("{{schema}}", schema_str)

        self.sub_agent = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=_JP_SUB_AGENT_TEMPLATE,
        )
        self.synthesizer = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=jp_synthesis,
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

        # 日足データ: yfinanceが失敗した場合はStooqにフォールバック
        if isinstance(daily, Exception) or not daily:
            logger.warning("yfinance get_daily failed for %s — trying Stooq fallback", ticker)
            daily = await StooqClient().get_daily(ticker, period="10d")

        # Daily bars
        if isinstance(daily, Exception) or not daily:
            daily_summary = "【日足データ】取得失敗 — データなし"
        else:
            rows = json.dumps(daily[:10], ensure_ascii=False)[:2000]
            daily_summary = f"【日足データ（直近10営業日）】\n{rows}"

        # Fundamentals
        if isinstance(info, Exception) or not info:
            funda_summary = "【企業情報】取得失敗 — データなし"
        else:
            rows = json.dumps(info, ensure_ascii=False)[:1500]
            funda_summary = f"【企業情報・財務指標】\n{rows}"

        # News
        if isinstance(news, Exception) or not news:
            news_summary = "【ニュース】取得失敗 — データなし"
        else:
            rows = json.dumps(news[:5], ensure_ascii=False)[:800]
            news_summary = f"【最新ニュース】\n{rows}"

        combined = f"{daily_summary}\n\n{funda_summary}\n\n{news_summary}"

        return {
            AgentRole.FUNDAMENTAL_ANALYST: combined,
            AgentRole.TECHNICAL_ANALYST: combined,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="7203.T", help="TSE ticker e.g. 7203.T (Toyota)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    agent = JapanAgent()
    thesis = await agent.analyze(args.ticker)
    print(thesis.model_dump_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(_main())
