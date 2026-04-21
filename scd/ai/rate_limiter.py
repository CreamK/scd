from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Token-bucket rate limiter for API calls."""

    def __init__(self, max_concurrent: int = 10, requests_per_minute: int = 60) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._interval = 60.0 / requests_per_minute
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_request_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_time = time.monotonic()

    def release(self) -> None:
        self._semaphore.release()

    async def __aenter__(self) -> RateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.release()
