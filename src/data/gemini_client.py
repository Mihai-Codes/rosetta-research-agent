"""Thin AdalFlow-compatible ModelClient wrapping the new google-genai SDK.

AdalFlow's built-in GoogleGenAIClient uses the deprecated `google.generativeai`
package which no longer works reliably. This wrapper uses `google.genai` (the
new official SDK) and exposes the same interface expected by adal.Generator.

Usage (drop-in for GoogleGenAIClient):
    from data.gemini_client import GeminiClient
    client = GeminiClient(api_key=os.getenv("GEMINI_API_KEY"))
"""

from __future__ import annotations

import logging
import os
from typing import Any

from adalflow.core.model_client import ModelClient

logger = logging.getLogger(__name__)


class GeminiClient(ModelClient):
    """Minimal synchronous AdalFlow ModelClient backed by google.genai."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY must be set")

    # ------------------------------------------------------------------
    # AdalFlow ModelClient interface — only the parts Generator calls
    # ------------------------------------------------------------------

    def convert_inputs_to_api_kwargs(
        self,
        input: Any = None,
        model_kwargs: dict[str, Any] | None = None,
        model_type: Any = None,
    ) -> dict[str, Any]:
        """Merge prompt + model_kwargs into a single api_kwargs dict."""
        result = dict(model_kwargs or {})
        result["prompt"] = input
        return result

    def call(
        self,
        api_kwargs: dict[str, Any] | None = None,
        model_type: Any = None,
    ) -> Any:
        """Call Gemini synchronously.

        Returns the raw response object — AdalFlow Generator calls
        parse_chat_completion() on this result to produce a GeneratorOutput.
        We return a plain dict with 'text' so parse_chat_completion can extract it.
        """
        from google import genai as google_genai

        kwargs = dict(api_kwargs or {})
        model  = kwargs.pop("model", "gemini-2.5-flash")
        prompt = kwargs.pop("prompt", "")
        config_kwargs: dict[str, Any] = {}
        if "temperature" in kwargs:
            config_kwargs["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            config_kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
        if "max_output_tokens" in kwargs:
            config_kwargs["max_output_tokens"] = kwargs.pop("max_output_tokens")
        kwargs.clear()

        client = google_genai.Client(api_key=self._api_key)
        cfg = google_genai.types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        call_kwargs: dict[str, Any] = {"model": model, "contents": prompt}
        if cfg is not None:
            call_kwargs["config"] = cfg
        # Let exceptions propagate — Generator catches them and wraps in GeneratorOutput
        response = client.models.generate_content(**call_kwargs)
        return response  # raw response; parse_chat_completion extracts .text

    async def acall(
        self,
        api_kwargs: dict[str, Any] | None = None,
        model_type: Any = None,
    ) -> Any:
        """Async call — runs blocking call in thread pool."""
        import asyncio
        return await asyncio.to_thread(self.call, api_kwargs, model_type)

    def parse_chat_completion(self, completion: Any) -> Any:
        """Extract text from Gemini response.

        Mirrors OpenAIClient convention: return GeneratorOutput(data=None,
        raw_response=text) so AdalFlow Generator picks up raw_response and
        runs it through its output_processors to populate data.
        """
        from adalflow.core.types import GeneratorOutput
        try:
            text = completion.text if hasattr(completion, "text") else str(completion)
            return GeneratorOutput(data=None, error=None, raw_response=text)
        except Exception as exc:
            logger.error("parse_chat_completion failed: %s", exc)
            return GeneratorOutput(data=None, error=str(exc), raw_response=str(completion))

    def track_completion_usage(self, completion: Any) -> None:
        return None
