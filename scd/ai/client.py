from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import anthropic
import httpx
from aiolimiter import AsyncLimiter

from scd.config import ScdConfig

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
JSON_PARSE_RETRIES = 2
INITIAL_BACKOFF = 3.0
API_TIMEOUT_SECONDS = 180.0


def _extract_text_from_content(content: list) -> str:
    """Extract text from response content blocks, skipping ThinkingBlock etc."""
    for block in content:
        if hasattr(block, "type") and block.type == "text" and hasattr(block, "text"):
            return block.text
    return ""


class ClaudeClient:
    """Async wrapper around the Anthropic API with rate limiting and retries."""

    def __init__(self, config: ScdConfig) -> None:
        kwargs: dict = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        kwargs["max_retries"] = 0
        kwargs["timeout"] = httpx.Timeout(
            connect=API_TIMEOUT_SECONDS,
            read=API_TIMEOUT_SECONDS,
            write=API_TIMEOUT_SECONDS,
            pool=API_TIMEOUT_SECONDS,
        )
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._model = config.model
        self._rate_limiter = AsyncLimiter(max_rate=config.rps, time_period=1.0)
        logger.info("Rate limiter: rps=%.1f", config.rps)
        self.total_calls = 0
        self._lock = asyncio.Lock()

    async def ask_json(self, system: str, user: str, max_tokens: int = 8192) -> dict:
        """Send a prompt and parse the response as JSON, with auto-retry on parse failure."""
        messages = [{"role": "user", "content": user}]

        for attempt in range(1 + JSON_PARSE_RETRIES):
            response = await self._api_call(
                system=system, messages=messages, max_tokens=max_tokens,
            )
            text = _extract_text_from_content(response.content) if response.content else ""
            try:
                return self._extract_json(text)
            except (json.JSONDecodeError, ValueError):
                if attempt >= JSON_PARSE_RETRIES:
                    raise
                logger.warning(
                    "JSON parse failed (attempt %d/%d), asking model to fix",
                    attempt + 1, 1 + JSON_PARSE_RETRIES,
                )
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": "Your previous response was not valid JSON. "
                    "Please respond with ONLY valid JSON, no markdown fences, no explanation.",
                })

    async def _api_call(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 8192,
    ) -> Any:
        """Make a single API call with retries."""
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                async with self._rate_limiter:
                    response = await self._client.messages.create(
                        model=self._model,
                        max_tokens=max_tokens,
                        system=system,
                        messages=messages,
                    )
                async with self._lock:
                    self.total_calls += 1
                return response
            except anthropic.RateLimitError:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning("Rate limited, retrying in %.1fs (attempt %d/%d)", backoff, attempt + 1, MAX_RETRIES)
                await asyncio.sleep(backoff)
            except anthropic.APIConnectionError as e:
                last_error = e
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                cause = repr(e.__cause__) if e.__cause__ else str(e)
                logger.warning("Connection error: %s, retrying in %.1fs (attempt %d/%d)", cause, backoff, attempt + 1, MAX_RETRIES)
                await asyncio.sleep(backoff)
            except anthropic.APIError as e:
                last_error = e
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning("API error: %s, retrying in %.1fs (attempt %d/%d)", e, backoff, attempt + 1, MAX_RETRIES)
                await asyncio.sleep(backoff)

        raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_error}")

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from response text, handling markdown blocks and mixed content."""
        text = text.strip()
        if not text:
            logger.warning("Empty response from API, returning empty dict")
            return {}

        # 1) Strip markdown code fences
        fence_match = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n\s*```", text)
        if fence_match:
            text = fence_match.group(1).strip()

        # 2) Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 3) Find the first { ... } or [ ... ] block in the text
        for opener, closer in [("{", "}"), ("[", "]")]:
            start = text.find(opener)
            if start == -1:
                continue
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break

        logger.error("Failed to extract JSON from response: %s", text[:500])
        raise ValueError(f"Cannot extract JSON from response: {text[:200]}")
