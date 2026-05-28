"""China regional agent — A-share equity analysis in Mandarin.

Data sources:
- Tushare Pro (daily OHLCV, financial indicators)

Model routing (per AGENTS.md):
- DeepSeek V3 / deepseek-chat — Chinese-language reasoning at cost-efficient rates.
  Prompts are intentionally in Simplified Chinese so DeepSeek uses its strongest
  reasoning path. The output InvestmentThesis is still English-schema Pydantic.

Run standalone:
    TUSHARE_TOKEN=xxx DEEPSEEK_API_KEY=xxx uv run python -m agents.china_agent --ticker 600519.SH
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

import adalflow as adal
from data.deepseek_client import DeepSeekClient

from agents.base_agent import PydanticJsonParser, RegionalAgent
from data.tushare_client import TushareClient
from reasoning.trace_schema import (
    AgentRole,
    AssetClass,
    InvestmentThesis,
    ReasoningBlock,
    Region,
)

logger = logging.getLogger(__name__)

# DeepSeek model override — see data/deepseek_client.py for the client impl.
# Override at runtime with DEEPSEEK_MODEL env var (e.g. "deepseek-reasoner" for R1).
# Default: deepseek-chat → maps to DeepSeek-V3 (latest stable, best CN reasoning).
_DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# ---------------------------------------------------------------------------
# Chinese-language prompt templates
# ---------------------------------------------------------------------------

_CN_SUB_AGENT_TEMPLATE = """\
<SYS>
你是一位专注于A股市场的专业分析师，角色为：{{role}}。
请根据以下市场数据对股票进行深入分析，重点关注盈利能力、估值水平、市场情绪及政策影响。
你需要展示深度思考的过程（类似R1的思维链），然后给出最终的分析和英文翻译。

输出必须是严格的JSON格式，不要有任何额外文字，字段名必须完全匹配：
{"agent_role": "{{role}}", "input_data_summary": "<50字以内：输入数据摘要>", "thought_process": "<详细的深度思考过程，至少200字，展示逻辑推演、正反面论证和数据支持>", "analysis": "<最终的分析结论，100-200字>", "analysis_en": "<English translation of the final analysis>", "conclusion": "<one sentence>", "confidence": <0.0-1.0>, "language": "zh"}
</SYS>

【股票】{{ticker}} | 【市场】{{region}} | 【类别】{{asset_class}}

{{data_summary}}\
"""

_CN_SYNTHESIS_TEMPLATE = """\
你是一位资深的A股组合经理，负责综合各位分析师的意见，形成最终投资建议。

【股票】{{ticker}} | 【市场】{{region}} | 【类别】{{asset_class}}
{% if learned_guidelines %}
=== 学习指南（永久应用）===
{{learned_guidelines}}
=== 指南结束 ===
{% endif %}
{% if prior_feedback %}
=== 上轮优化反馈 ===
{{prior_feedback}}
=== 反馈结束 ===
{% endif %}
分析师报告汇总：
{{analyst_reports}}

只输出一个合法的JSON对象，不要有任何前缀或解释。字段要求如下：
- asset_class: "equity"
- ticker_or_asset: 股票代码字符串
- thesis_summary_en: 英文摘要（必填）
- thesis_summary_native: 中文摘要（必填）
- direction: "LONG" 或 "SHORT" 或 "NEUTRAL"（大写英文）
- confidence_score: 0.0到1.0的浮点数
- time_horizon_days: 正整数（建议持有天数）
- reasoning_blocks: []
- data_sources_used: 数据来源列表
- risk_factors: 风险因素列表
- model_routing: {}
\
"""


class ChinaAgent(RegionalAgent):
    """A-share analyst — runs all reasoning in Simplified Chinese via DeepSeek."""

    region = Region.CN
    working_language = "zh"
    sub_agent_roles = (
        AgentRole.FUNDAMENTAL_ANALYST,
        AgentRole.SENTIMENT_ANALYST,
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
        # Build DeepSeek client; fall back to Groq if no DEEPSEEK_API_KEY is set.
        if model_client is None:
            deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
            _use_groq = not deepseek_key
            if deepseek_key and not _use_groq:
                # Probe DeepSeek to detect 402 (Insufficient Balance) before full analysis.
                import httpx as _httpx
                try:
                    _r = _httpx.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        json={"model": _DEEPSEEK_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                        headers={"Authorization": f"Bearer {deepseek_key}"},
                        timeout=10,
                    )
                    if _r.status_code == 402:
                        logger.warning("DeepSeek 402 Insufficient Balance — falling back to Groq for CN desk")
                        _use_groq = True
                except Exception as _probe_exc:
                    logger.warning("DeepSeek probe failed (%s) — falling back to Groq for CN desk", _probe_exc)
                    _use_groq = True
            if _use_groq:
                # Try OpenRouter (DeepSeek V4 Flash, free tier) before falling back to Groq.
                openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
                if openrouter_key:
                    logger.info("CN desk: using OpenRouter DeepSeek V4 Flash (free tier)")
                    from data.deepseek_client import DeepSeekClient as _DSC
                    model_client = _DSC(
                        api_key=openrouter_key,
                        base_url="https://openrouter.ai/api/v1",
                    )
                    if model_kwargs is None:
                        model_kwargs = {"model": "deepseek/deepseek-v4-flash", "temperature": 0.2, "max_tokens": 2048}
                else:
                    logger.warning("Using Groq fallback for CN desk")
                    model_client = adal.GroqAPIClient()  # type: ignore[attr-defined]
                    if model_kwargs is None:
                        model_kwargs = {"model": "llama-3.3-70b-versatile", "temperature": 0.2, "max_tokens": 2048}
            else:
                model_client = DeepSeekClient(api_key=deepseek_key)
                if model_kwargs is None:
                    model_kwargs = {"model": _DEEPSEEK_MODEL, "temperature": 0.2, "max_tokens": 2048}
        if model_kwargs is None:
            model_kwargs = {"model": _DEEPSEEK_MODEL, "temperature": 0.2, "max_tokens": 2048}

        # Call parent — it initialises self._block_parser / self._thesis_parser.
        super().__init__(model_client=model_client, model_kwargs=model_kwargs)

        # Rebuild generators with Chinese-language templates.
        schema_str = json.dumps(
            InvestmentThesis.model_json_schema(), ensure_ascii=False, indent=2
        )
        cn_synthesis = _CN_SYNTHESIS_TEMPLATE.replace("{{schema}}", schema_str)

        self.sub_agent = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=_CN_SUB_AGENT_TEMPLATE,
        )
        self.synthesizer = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=cn_synthesis,
        )

        # Re-assign parsers (parent already created them; keep consistent refs).
        self._block_parser = PydanticJsonParser(ReasoningBlock)
        self._thesis_parser = PydanticJsonParser(InvestmentThesis)

    # ------------------------------------------------------------------
    # Data sources — Tushare
    # ------------------------------------------------------------------

    async def get_data_sources(self, ticker: str) -> dict[AgentRole, str]:
        """Pull daily bars + financial indicators.

        Priority:
        1. Tushare Pro (if TUSHARE_TOKEN is set and returns data)
        2. AKShare / Eastmoney (free, no token required) — automatic fallback
        """
        from data.akshare_client import get_cn_data_bundle, _tushare_to_yf

        daily: list = []
        fina: list = []
        info: dict = {}
        news: list = []
        source_used = "none"

        # --- Tushare (preferred, requires token) ---
        tushare_token = os.environ.get("TUSHARE_TOKEN", "").strip()
        if tushare_token:
            async with TushareClient() as ts:
                _daily, _fina = await asyncio.gather(
                    ts.get_daily(ticker, limit=10),
                    ts.get_fina_indicator(ticker),
                    return_exceptions=True,
                )
            if not isinstance(_daily, Exception) and _daily:
                daily = _daily
                source_used = "tushare"
            if not isinstance(_fina, Exception) and _fina:
                fina = _fina

        # --- AKShare fallback (free, Eastmoney backend) ---
        if not daily:
            logger.info("Tushare unavailable/empty for %s — trying AKShare", ticker)
            try:
                bundle = await get_cn_data_bundle(ticker)
                daily = bundle.get("daily", [])
                info = bundle.get("info", {})
                news = bundle.get("news", [])
                if daily:
                    source_used = "akshare"
                    logger.info("AKShare: got %d daily bars for %s", len(daily), ticker)
            except Exception as _ak_exc:
                logger.warning("AKShare bundle failed for %s: %s", ticker, _ak_exc)

        # --- yfinance price snapshot (correct symbol mapping) ---
        yf_ticker = _tushare_to_yf(ticker)  # 600519.SH → 600519.SS
        try:
            import yfinance as yf
            loop = asyncio.get_event_loop()
            yf_info = await loop.run_in_executor(
                None,
                lambda: yf.Ticker(yf_ticker).fast_info,
            )
            price_snapshot = {
                "last_price": getattr(yf_info, "last_price", None),
                "market_cap": getattr(yf_info, "market_cap", None),
                "52w_high": getattr(yf_info, "year_high", None),
                "52w_low": getattr(yf_info, "year_low", None),
            }
            logger.info("yfinance snapshot for %s: %s", yf_ticker, price_snapshot)
        except Exception as _yf_exc:
            logger.debug("yfinance snapshot skipped for %s: %s", yf_ticker, _yf_exc)
            price_snapshot = {}

        daily_summary = json.dumps(daily, ensure_ascii=False, default=str)[:3000] if daily else "[无日线数据]"
        fina_summary = json.dumps(fina, ensure_ascii=False, default=str)[:2000] if fina else "[无财务数据]"
        info_summary = json.dumps(info, ensure_ascii=False, default=str)[:1500] if info else "[无公司信息]"
        news_summary = json.dumps(news, ensure_ascii=False, default=str)[:1500] if news else "[无新闻数据]"
        price_summary = json.dumps(price_snapshot, ensure_ascii=False)[:500] if price_snapshot else "[无价格快照]"

        fundamental_data = (
            f"【股票代码】{ticker} (yfinance: {yf_ticker}) | 数据来源: {source_used}\n"
            f"【实时价格快照】\n{price_summary}\n\n"
            f"【近期日线数据（最近10交易日）】\n{daily_summary}\n\n"
            f"【财务指标（近4期）】\n{fina_summary}\n\n"
            f"【公司基本信息】\n{info_summary}"
        )
        sentiment_data = (
            f"【股票代码】{ticker} | 数据来源: {source_used}\n"
            f"【近期新闻】\n{news_summary}\n\n"
            f"【实时价格快照】\n{price_summary}"
        )

        return {
            AgentRole.FUNDAMENTAL_ANALYST: fundamental_data,
            AgentRole.SENTIMENT_ANALYST: sentiment_data,
        }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the China agent on an A-share ticker.")
    parser.add_argument("--ticker", default="600519.SH")  # Kweichow Moutai
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    agent = ChinaAgent()
    thesis = await agent.analyze(args.ticker)
    print(thesis.model_dump_json(indent=2, ensure_ascii=False))


def run() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    run()
