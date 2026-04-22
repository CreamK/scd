from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Sliding-window rate limiter that allows true concurrent bursts."""

    def __init__(self, max_concurrent: int = 10, requests_per_second: float = 2.0) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rps = requests_per_second
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        async with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < 1.0]

            if len(self._timestamps) >= self._rps:
                oldest = self._timestamps[0]
                wait = 1.0 - (now - oldest)
                if wait > 0:
                    await asyncio.sleep(wait)
                    now = time.monotonic()
                    self._timestamps = [t for t in self._timestamps if now - t < 1.0]

            self._timestamps.append(now)

    def release(self) -> None:
        self._semaphore.release()

    async def __aenter__(self) -> RateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.release()
