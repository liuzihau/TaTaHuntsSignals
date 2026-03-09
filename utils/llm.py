"""LLM client wrapper for Anthropic Claude calls and JSON parsing helpers."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from utils.trace_logger import log_llm_response

try:
    import anthropic
except Exception:  # pragma: no cover - handled via runtime errors in LLMClient
    anthropic = None


class LLMClient:
    """Small Anthropic client with raw and JSON invocation helpers."""

    def __init__(self, model: str = "claude-sonnet-4-0") -> None:
        load_dotenv()
        env_model = os.getenv("ANTHROPIC_MODEL", "").strip()
        self.model = env_model or model
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing. Add it to your environment or .env.")
        if anthropic is None:
            raise ImportError("anthropic package is not installed. Add anthropic>=0.25.0.")
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def _candidate_models(self) -> list[str]:
        candidates: list[str] = []
        env_model = os.getenv("ANTHROPIC_MODEL", "").strip()
        if env_model:
            candidates.append(env_model)

        candidates.extend(
            [
                self.model,
                "claude-sonnet-4-0",
                "claude-sonnet-4-20250514",
                "claude-3-7-sonnet-latest",
                "claude-3-7-sonnet-20250219",
                "claude-3-5-sonnet-latest",
            ]
        )
        # Deduplicate while preserving order.
        return list(dict.fromkeys([m for m in candidates if m]))

    @staticmethod
    def _is_model_not_found_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "not_found_error" in message or "model" in message and "not found" in message

    def _invoke_with_model(self, prompt: str, model: str) -> str:
        response = self._client.messages.create(
            model=model,
            max_tokens=700,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        blocks = getattr(response, "content", [])
        texts: list[str] = []
        for block in blocks:
            if getattr(block, "type", None) == "text":
                texts.append(getattr(block, "text", ""))
        return "\n".join(texts).strip()

    def invoke(self, prompt: str) -> str:
        """Call Claude and return raw text content."""
        last_error: Exception | None = None
        for model in self._candidate_models():
            try:
                output = self._invoke_with_model(prompt, model)
                self.model = model
                log_llm_response(
                    {
                        "model": model,
                        "status": "success",
                        "prompt": prompt,
                        "response": output,
                    }
                )
                return output
            except Exception as exc:
                last_error = exc
                log_llm_response(
                    {
                        "model": model,
                        "status": "error",
                        "prompt": prompt,
                        "error": str(exc),
                    }
                )
                if self._is_model_not_found_error(exc):
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("No Anthropic model candidates available to invoke.")

    def invoke_json(self, prompt: str) -> dict[str, Any]:
        """Call Claude and parse dict-like JSON, with regex fallback extraction."""
        raw = self.invoke(prompt)

        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {}

        extracted = match.group(0).strip()
        try:
            parsed = json.loads(extracted)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
