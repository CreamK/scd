from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable, Coroutine

import anthropic

from scd.ai.rate_limiter import RateLimiter
from scd.config import ScdConfig

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
JSON_PARSE_RETRIES = 2
INITIAL_BACKOFF = 2.0

ToolHandler = Callable[[str, dict], Coroutine[Any, Any, str]]


def _extract_text_from_content(content: list) -> str:
    """Extract text from response content blocks, skipping ThinkingBlock etc."""
    for block in content:
        if hasattr(block, "type") and block.type == "text" and hasattr(block, "text"):
            return block.text
    return ""


class ClaudeClient:
    """Async wrapper around the Anthropic API with rate limiting, retries, and tool use."""

    def __init__(self, config: ScdConfig) -> None:
        kwargs: dict = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._model = config.model
        self._rate_limiter = RateLimiter(
            max_concurrent=config.concurrency,
            requests_per_minute=config.concurrency * 6,
        )
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

    async def ask(self, system: str, user: str, max_tokens: int = 8192) -> str:
        """Send a prompt and return raw text response."""
        response = await self._api_call(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        return _extract_text_from_content(response.content)

    async def agent_loop(
        self,
        system: str,
        user: str,
        tools: list[dict],
        tool_handler: ToolHandler,
        max_tokens: int = 8192,
        max_turns: int = 50,
    ) -> str:
        """Run a multi-turn agent loop with tool use.

        Claude calls tools, we execute them and feed results back,
        until Claude produces a final text response (stop_reason='end_turn').
        """
        messages: list[dict] = [{"role": "user", "content": user}]

        for turn in range(max_turns):
            response = await self._api_call(
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                tools=tools,
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        return block.text
                return ""

            if response.stop_reason != "tool_use":
                for block in response.content:
                    if block.type == "text":
                        return block.text
                return ""

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    logger.debug("Tool call: %s(%s)", block.name, json.dumps(block.input, ensure_ascii=False)[:200])
                    try:
                        result = await tool_handler(block.name, block.input)
                    except Exception as e:
                        result = f"Error: {e}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})

        logger.warning("Agent loop reached max turns (%d)", max_turns)
        return ""

    async def agent_loop_json(
        self,
        system: str,
        user: str,
        tools: list[dict],
        tool_handler: ToolHandler,
        max_tokens: int = 8192,
        max_turns: int = 50,
    ) -> dict:
        """Run agent loop and parse final response as JSON."""
        text = await self.agent_loop(system, user, tools, tool_handler, max_tokens, max_turns)
        return self._extract_json(text)

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
