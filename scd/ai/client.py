from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
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

    async def ask_json_with_tools(
        self,
        system: str,
        user: str,
        tools: list[dict],
        tool_handler: Callable[[str, dict], Awaitable[Any]],
        *,
        max_tool_turns: int = 20,
        validator: Callable[[dict], tuple[bool, str | None]] | None = None,
        max_tokens: int = 8192,
    ) -> dict:
        """Tool-use loop: dispatch tool_use blocks to tool_handler, parse final
        JSON on end_turn, optionally run validator and loop with a follow-up
        user message if validator rejects the result.

        - Each messages.create call counts against max_tool_turns.
        - tool_handler must return something JSON-serializable; it will be
          wrapped into a tool_result content block.
        - Raises RuntimeError when max_tool_turns is exhausted without a
          validated final answer.
        """
        messages: list[dict] = [{"role": "user", "content": user}]

        for turn in range(max_tool_turns):
            response = await self._api_call(
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                tools=tools,
            )
            stop_reason = getattr(response, "stop_reason", None)
            content = response.content or []

            if stop_reason == "tool_use":
                assistant_blocks: list[dict] = []
                tool_results: list[dict] = []
                for block in content:
                    btype = getattr(block, "type", None)
                    if btype == "text":
                        assistant_blocks.append(
                            {"type": "text", "text": getattr(block, "text", "") or ""}
                        )
                    elif btype == "tool_use":
                        tool_use_id = getattr(block, "id", "")
                        name = getattr(block, "name", "")
                        tool_input = getattr(block, "input", {}) or {}
                        assistant_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tool_use_id,
                                "name": name,
                                "input": tool_input,
                            }
                        )
                        try:
                            result = await tool_handler(name, tool_input)
                        except Exception as e:
                            logger.warning("tool_handler(%s) raised: %s", name, e)
                            result = {"error": f"{type(e).__name__}: {e}"}
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": json.dumps(result, ensure_ascii=False),
                            }
                        )
                if assistant_blocks:
                    messages.append({"role": "assistant", "content": assistant_blocks})
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                continue

            text = _extract_text_from_content(content)
            try:
                result = self._extract_json(text)
            except (json.JSONDecodeError, ValueError):
                if turn + 1 >= max_tool_turns:
                    raise
                logger.warning(
                    "JSON parse failed on tool-use end_turn (turn %d/%d), asking model to fix",
                    turn + 1, max_tool_turns,
                )
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": "Your previous response was not valid JSON. "
                        "Respond with ONLY valid JSON, no markdown fences, no explanation.",
                    }
                )
                continue

            if validator is None:
                return result
            ok, follow_up = validator(result)
            if ok:
                return result
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": follow_up
                    or "Your previous answer was rejected. Please revise and output final JSON.",
                }
            )

        raise RuntimeError(
            f"ask_json_with_tools exhausted max_tool_turns={max_tool_turns} "
            "without a validated final answer"
        )

    async def _api_call(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 8192,
        tools: list[dict] | None = None,
    ) -> Any:
        """Make a single API call with retries."""
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                async with self._rate_limiter:
                    kwargs: dict[str, Any] = {
                        "model": self._model,
                        "max_tokens": max_tokens,
                        "system": system,
                        "messages": messages,
                    }
                    if tools:
                        kwargs["tools"] = tools
                    response = await self._client.messages.create(**kwargs)
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
