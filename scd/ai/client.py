from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx
from aiolimiter import AsyncLimiter
from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)

from scd.config import ScdConfig

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
JSON_PARSE_RETRIES = 2
INITIAL_BACKOFF = 3.0
API_TIMEOUT_SECONDS = 300.0


class LlmClient:
    """Async wrapper around an OpenAI-compatible Chat Completions endpoint.

    Handles rate limiting, retries with backoff, and capability downgrade for
    self-hosted gateways that only partially implement the OpenAI spec
    (response_format, parallel_tool_calls, tool_choice).
    """

    def __init__(self, config: ScdConfig) -> None:
        kwargs: dict[str, Any] = {
            "max_retries": 0,
            "timeout": httpx.Timeout(
                connect=API_TIMEOUT_SECONDS,
                read=API_TIMEOUT_SECONDS,
                write=API_TIMEOUT_SECONDS,
                pool=API_TIMEOUT_SECONDS,
            ),
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = AsyncOpenAI(**kwargs)
        self._model = config.model
        self._rate_limiter = AsyncLimiter(max_rate=config.rps, time_period=1.0)
        max_in_flight = max(1, int(getattr(config, "max_in_flight", 8) or 8))
        self._inflight = asyncio.Semaphore(max_in_flight)
        logger.info(
            "Rate limiter: rps=%.1f, max_in_flight=%d",
            config.rps, max_in_flight,
        )
        self.total_calls = 0
        self._lock = asyncio.Lock()
        # Capability flags. Start from user config; auto-downgrade on 400.
        self._caps: dict[str, bool] = {
            "json_mode": bool(config.use_json_mode),
            "parallel_tool_calls": bool(config.parallel_tool_calls),
            "tool_choice": True,
        }

    async def ask_json(self, system: str, user: str, max_tokens: int = 8192) -> dict:
        """Send a prompt and parse the response as JSON, with auto-retry on parse failure."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        for attempt in range(1 + JSON_PARSE_RETRIES):
            resp = await self._chat(
                messages, max_tokens=max_tokens, want_json=True,
            )
            text = self._extract_message_text(resp) or ""
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

    async def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 8192,
        tools: list[dict] | None = None,
        want_json: bool = False,
    ) -> Any:
        """Low-level Chat Completions call with retries and capability downgrade."""
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                async with self._inflight, self._rate_limiter:
                    kwargs: dict[str, Any] = {
                        "model": self._model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                    }
                    if tools:
                        kwargs["tools"] = tools
                        if self._caps["tool_choice"]:
                            kwargs["tool_choice"] = "auto"
                        if self._caps["parallel_tool_calls"]:
                            kwargs["parallel_tool_calls"] = True
                    if want_json and self._caps["json_mode"] and not tools:
                        kwargs["response_format"] = {"type": "json_object"}
                    resp = await self._client.chat.completions.create(**kwargs)
                async with self._lock:
                    self.total_calls += 1
                return resp
            except RateLimitError:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    "Rate limited, retrying in %.1fs (attempt %d/%d)",
                    backoff, attempt + 1, MAX_RETRIES,
                )
                await asyncio.sleep(backoff)
            except BadRequestError as e:
                # Self-hosted gateways often reject optional fields -> downgrade once.
                emsg = (str(e) or "").lower()
                downgraded = False
                if "response_format" in emsg and self._caps["json_mode"]:
                    logger.warning(
                        "Endpoint rejected response_format; disabling json_mode "
                        "for the rest of this session"
                    )
                    self._caps["json_mode"] = False
                    downgraded = True
                if "parallel_tool_calls" in emsg and self._caps["parallel_tool_calls"]:
                    logger.warning(
                        "Endpoint rejected parallel_tool_calls; disabling for the "
                        "rest of this session"
                    )
                    self._caps["parallel_tool_calls"] = False
                    downgraded = True
                if "tool_choice" in emsg and self._caps["tool_choice"]:
                    logger.warning(
                        "Endpoint rejected tool_choice; disabling for the rest of "
                        "this session"
                    )
                    self._caps["tool_choice"] = False
                    downgraded = True
                if not downgraded:
                    raise
                # Retry immediately with the downgraded capability.
            except APIConnectionError as e:
                last_error = e
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                cause = repr(e.__cause__) if e.__cause__ else str(e)
                logger.warning(
                    "Connection error: %s, retrying in %.1fs (attempt %d/%d)",
                    cause, backoff, attempt + 1, MAX_RETRIES,
                )
                await asyncio.sleep(backoff)
            except APIError as e:
                last_error = e
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    "API error: %s, retrying in %.1fs (attempt %d/%d)",
                    e, backoff, attempt + 1, MAX_RETRIES,
                )
                await asyncio.sleep(backoff)

        raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_error}")

    @staticmethod
    def _extract_message_text(resp: Any) -> str:
        """Pull out assistant text from a Chat Completions response.

        Robust to providers that return either a plain string or a list of
        content parts (some OpenAI-compatible servers do the latter).
        """
        try:
            msg = resp.choices[0].message
        except (AttributeError, IndexError):
            return ""
        content = getattr(msg, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                else:
                    text = getattr(block, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

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
