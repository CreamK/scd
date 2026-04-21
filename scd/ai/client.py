from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine

import anthropic

from scd.ai.rate_limiter import RateLimiter
from scd.config import ScdConfig

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0

ToolHandler = Callable[[str, dict], Coroutine[Any, Any, str]]


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
        """Send a prompt and parse the response as JSON."""
        text = await self.ask(system, user, max_tokens)
        return self._extract_json(text)

    async def ask(self, system: str, user: str, max_tokens: int = 8192) -> str:
        """Send a prompt and return raw text response."""
        response = await self._api_call(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        text = response.content[0].text
        return text

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
        """Extract JSON from response text, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            start = 1
            end = len(lines) - 1
            if lines[end].strip() == "```":
                pass
            else:
                end = len(lines)
            text = "\n".join(lines[start:end]).strip()

        return json.loads(text)
