"""Pair-level checkpoint cache for Phase 3 (function comparison).

Uses an append-only JSONL file so that interrupted runs (e.g. network failures)
can resume without losing already-computed results. Each completed file pair is
flushed to disk immediately after the AI call returns.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

from scd.models import (
    CompareResult,
    DimensionScores,
    FuncLocation,
    SimilarFunction,
    SimilarityLevel,
)

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".scd_cache"
CACHE_FILE_NAME = "pair_results.jsonl"
CACHE_VERSION = 1


def compute_pair_key(
    file_a: str,
    content_a: str,
    file_b: str,
    content_b: str,
    model: str,
    threshold: int,
) -> str:
    """Stable key for a (file_a, file_b) comparison under given model/threshold."""
    h = hashlib.sha256()
    h.update(f"v{CACHE_VERSION}".encode())
    h.update(b"\0")
    h.update(model.encode())
    h.update(b"\0")
    h.update(str(threshold).encode())
    h.update(b"\0")
    h.update(file_a.encode())
    h.update(b"\0")
    h.update(content_a.encode())
    h.update(b"\0")
    h.update(file_b.encode())
    h.update(b"\0")
    h.update(content_b.encode())
    return h.hexdigest()[:16]


def _cache_path(output_dir: str) -> Path:
    return Path(output_dir) / CACHE_DIR_NAME / CACHE_FILE_NAME


def _result_to_record(key: str, result: CompareResult) -> dict:
    return {
        "v": CACHE_VERSION,
        "key": key,
        "file_a": result.file_a,
        "file_b": result.file_b,
        "similar_functions": [
            {
                "func_a": asdict(sf.func_a),
                "func_b": asdict(sf.func_b),
                "composite_score": sf.composite_score,
                "similarity_level": sf.similarity_level.value,
                "scores": asdict(sf.scores),
                "analysis": sf.analysis,
            }
            for sf in result.similar_functions
        ],
    }


def _record_to_result(data: dict) -> CompareResult:
    similar: list[SimilarFunction] = []
    for sf in data.get("similar_functions", []):
        similar.append(
            SimilarFunction(
                func_a=FuncLocation(**sf["func_a"]),
                func_b=FuncLocation(**sf["func_b"]),
                composite_score=int(sf["composite_score"]),
                similarity_level=SimilarityLevel(sf["similarity_level"]),
                scores=DimensionScores(**sf["scores"]),
                analysis=sf.get("analysis", ""),
            )
        )
    return CompareResult(
        file_a=data["file_a"],
        file_b=data["file_b"],
        similar_functions=similar,
    )


class PairCache:
    """Append-only JSONL cache of pair comparison results."""

    def __init__(self, output_dir: str) -> None:
        self._path = _cache_path(output_dir)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, CompareResult] = {}
        self._lock = asyncio.Lock()

    def load(self) -> int:
        """Load existing checkpoint entries from disk. Returns count loaded."""
        if not self._path.exists():
            return 0
        count = 0
        malformed = 0
        with self._path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    key = data["key"]
                    self._store[key] = _record_to_result(data)
                    count += 1
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    malformed += 1
                    logger.warning(
                        "Skipping malformed cache line %d in %s: %s",
                        line_no, self._path, e,
                    )
        if malformed:
            logger.warning("%d malformed lines skipped in pair cache", malformed)
        return count

    def get(self, key: str) -> CompareResult | None:
        return self._store.get(key)

    async def put(self, key: str, result: CompareResult) -> None:
        """Append a completed result to the cache (idempotent)."""
        async with self._lock:
            if key in self._store:
                return
            self._store[key] = result
            record = _result_to_record(key, result)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    @property
    def path(self) -> Path:
        return self._path
