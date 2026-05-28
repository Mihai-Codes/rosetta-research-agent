"""AdalFlow-compatible ModelClient for DeepSeek API.

AdalFlow's built-in OpenAIClient now uses the new OpenAI Responses API
(/v1/responses) which DeepSeek does not support. This wrapper calls
/v1/chat/completions directly via the openai SDK's legacy interface.

Usage:
    from data.deepseek_client import DeepSeekClient
    client = DeepSeekClient(api_key=os.getenv("DEEPSEEK_API_KEY"))
"""

from __future__ import annotations

import logging
import os
from typing import Any

from adalflow.core.model_client import ModelClient
from adalflow.core.types import GeneratorOutput

logger = logging.getLogger(__name__)

_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekClient(ModelClient):
    """Thin AdalFlow ModelClient calling DeepSeek via chat.completions."""

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self._api_key:
            raise ValueError("DEEPSEEK_API_KEY must be set")

    def convert_inputs_to_api_kwargs(
        self,
        input: Any = None,
        model_kwargs: dict[str, Any] | None = None,
        model_type: Any = None,
    ) -> dict[str, Any]:
        result = dict(model_kwargs or {})
        result["messages"] = [{"role": "user", "content": input}]
        return result

    def call(
        self,
        api_kwargs: dict[str, Any] | None = None,
        model_type: Any = None,
    ) -> Any:
        """Call DeepSeek using chat.completions (not the new Responses API)."""
        from openai import OpenAI

        kwargs = dict(api_kwargs or {})
        client = OpenAI(api_key=self._api_key, base_url=_DEEPSEEK_BASE_URL)
        # Call chat.completions directly — DeepSeek doesn't support /v1/responses
        response = client.chat.completions.create(**kwargs)
        return response

    async def acall(
        self,
        api_kwargs: dict[str, Any] | None = None,
        model_type: Any = None,
    ) -> Any:
        import asyncio
        return await asyncio.to_thread(self.call, api_kwargs, model_type)

    def parse_chat_completion(self, completion: Any) -> GeneratorOutput:
        """Extract content from ChatCompletion → GeneratorOutput."""
        try:
            text = completion.choices[0].message.content
            return GeneratorOutput(data=None, error=None, raw_response=text)
        except Exception as exc:
            logger.error("DeepSeekClient parse_chat_completion failed: %s", exc)
            return GeneratorOutput(data=None, error=str(exc), raw_response=str(completion))

    def track_completion_usage(self, completion: Any) -> None:
        return None
