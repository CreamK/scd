from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict
from itertools import product
from pathlib import Path

from scd.ai.client import LlmClient
from scd.ai.prompts import FUNCTION_COMPARE_SYSTEM, FUNCTION_COMPARE_USER
from scd.config import ScdConfig
from scd.models import (
    CompareResult,
    DimensionScores,
    DirMatch,
    FuncLocation,
    RepoScanResult,
    SimilarFunction,
    SimilarityLevel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pair-level checkpoint cache
#
# Append-only JSONL file so that interrupted runs (e.g. network failures) can
# resume without losing already-computed results. Each completed file pair is
# flushed to disk immediately after the AI call returns.
# ---------------------------------------------------------------------------

PAIR_CACHE_DIR_NAME = ".scd_cache"
PAIR_CACHE_FILE_NAME = "pair_results.jsonl"
PAIR_CACHE_VERSION = 1


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
    h.update(f"v{PAIR_CACHE_VERSION}".encode())
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


def _pair_cache_path(output_dir: str) -> Path:
    return Path(output_dir) / PAIR_CACHE_DIR_NAME / PAIR_CACHE_FILE_NAME


def _result_to_record(key: str, result: CompareResult) -> dict:
    return {
        "v": PAIR_CACHE_VERSION,
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
        self._path = _pair_cache_path(output_dir)
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


def _iter_subtree_files(repo: RepoScanResult, dir_path: str) -> list[str]:
    """Collect paths of every source file under `dir_path` (inclusive, recursive).

    `dir_path == ""` is the repo root, i.e. the whole repo.
    """
    prefix = f"{dir_path}/" if dir_path else ""
    paths: list[str] = []
    for d, dir_info in repo.dirs.items():
        if dir_path and d != dir_path and not d.startswith(prefix):
            continue
        for f in dir_info.files:
            paths.append(f.path)
    return paths


def _build_file_pairs(
    match: DirMatch,
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
) -> list[tuple[str, str]]:
    """Generate all file pairs (n x m) for a matched directory pair.

    A matched directory claims every file under its subtree on both sides, so
    files in nested subdirectories also participate in comparison. Cross-match
    duplicates are removed later by ``build_all_file_pairs``.
    """
    if match.dir_a not in repo_a.dirs or match.dir_b not in repo_b.dirs:
        return []

    files_a = _iter_subtree_files(repo_a, match.dir_a)
    files_b = _iter_subtree_files(repo_b, match.dir_b)
    return list(product(files_a, files_b))


def _parse_similar_functions(data: dict, file_a: str, file_b: str) -> list[SimilarFunction]:
    """Parse AI response into SimilarFunction objects."""
    results: list[SimilarFunction] = []
    for item in data.get("similar_functions", []):
        try:
            fa = item["func_a"]
            fb = item["func_b"]

            raw_scores = item.get("scores", {})
            scores = DimensionScores(
                data_structure=int(raw_scores.get("data_structure", 0)),
                function_signature=int(raw_scores.get("function_signature", 0)),
                algorithm_logic=int(raw_scores.get("algorithm_logic", 0)),
                naming_convention=int(raw_scores.get("naming_convention", 0)),
                protocol_conformance=int(raw_scores.get("protocol_conformance", 0)),
            )

            composite = int(item.get("composite_score", 0))

            level_str = item.get("similarity_level", "very_low")
            try:
                level = SimilarityLevel(level_str)
            except ValueError:
                level = SimilarityLevel.VERY_LOW

            results.append(SimilarFunction(
                func_a=FuncLocation(
                    file=file_a,
                    name=fa.get("name", "unknown"),
                    line_start=fa.get("line_start", 0),
                    line_end=fa.get("line_end", 0),
                ),
                func_b=FuncLocation(
                    file=file_b,
                    name=fb.get("name", "unknown"),
                    line_start=fb.get("line_start", 0),
                    line_end=fb.get("line_end", 0),
                ),
                composite_score=composite,
                similarity_level=level,
                scores=scores,
                analysis=item.get("analysis", ""),
            ))
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Failed to parse similar function entry: %s", e)
    return results


async def _compare_file_pair(
    file_a: str,
    file_b: str,
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
    client: LlmClient,
    threshold: int,
    model: str,
    cache: PairCache | None = None,
    progress: dict | None = None,
) -> CompareResult:
    """Compare two files using the configured LLM, with optional checkpoint cache."""
    code_a = repo_a.file_contents.get(file_a, "")
    code_b = repo_b.file_contents.get(file_b, "")

    if not code_a or not code_b:
        if progress is not None:
            progress["skipped"] = progress.get("skipped", 0) + 1
        return CompareResult(file_a=file_a, file_b=file_b)

    cache_key: str | None = None
    if cache is not None:
        cache_key = compute_pair_key(file_a, code_a, file_b, code_b, model, threshold)
        cached = cache.get(cache_key)
        if cached is not None:
            if progress is not None:
                progress["cached"] = progress.get("cached", 0) + 1
            return cached

    system = FUNCTION_COMPARE_SYSTEM.format(threshold=threshold)
    user = FUNCTION_COMPARE_USER.format(
        file_a=file_a, file_b=file_b,
        code_a=code_a, code_b=code_b,
    )

    try:
        data = await client.ask_json(system, user)
        similar = _parse_similar_functions(data, file_a, file_b)
        result = CompareResult(file_a=file_a, file_b=file_b, similar_functions=similar)
        if cache is not None and cache_key is not None:
            await cache.put(cache_key, result)
        if progress is not None:
            progress["completed"] = progress.get("completed", 0) + 1
            done = progress["completed"] + progress.get("cached", 0) + progress.get("skipped", 0)
            total = progress.get("total", 0)
            interval = progress.get("log_every", 25)
            if total and (progress["completed"] % interval == 0 or done == total):
                logger.info(
                    "Phase 3 progress: %d/%d done (cached=%d, new=%d, skipped=%d)",
                    done, total,
                    progress.get("cached", 0),
                    progress["completed"],
                    progress.get("skipped", 0),
                )
        return result
    except Exception as e:
        logger.error("Error comparing %s <-> %s: %s", file_a, file_b, e)
        if progress is not None:
            progress["errors"] = progress.get("errors", 0) + 1
        return CompareResult(file_a=file_a, file_b=file_b)


def build_all_file_pairs(
    matched_dirs: list[DirMatch],
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
) -> list[tuple[str, str]]:
    """Build all file pairs from matched directory pairs, deduplicated.

    For each matched directory pair, every source file under the A-side subtree
    is paired with every source file under the B-side subtree. Pairs that
    appear in multiple matches (e.g. when matches nest) are kept once.
    Insertion order is preserved so output is deterministic.
    """
    seen: set[tuple[str, str]] = set()
    unique_pairs: list[tuple[str, str]] = []
    for match in matched_dirs:
        pairs = _build_file_pairs(match, repo_a, repo_b)
        new_count = 0
        for fa, fb in pairs:
            key = (fa, fb)
            if key in seen:
                continue
            seen.add(key)
            unique_pairs.append(key)
            new_count += 1
        logger.info(
            "Dir pair %s <-> %s: %d file pairs (subtree), %d new after dedup",
            match.dir_a or "(root)",
            match.dir_b or "(root)",
            len(pairs),
            new_count,
        )
    logger.info("Total unique file pairs to compare: %d", len(unique_pairs))
    return unique_pairs


async def compare_file_pairs(
    file_pairs: list[tuple[str, str]],
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
    client: LlmClient,
    config: ScdConfig,
    cache: PairCache | None = None,
) -> list[CompareResult]:
    """Compare a pre-built list of file pairs, reusing cached results when available."""
    progress: dict = {
        "total": len(file_pairs),
        "cached": 0,
        "completed": 0,
        "skipped": 0,
        "errors": 0,
        "log_every": 25,
    }

    tasks = [
        _compare_file_pair(
            fa, fb, repo_a, repo_b, client,
            config.similarity_threshold, config.model,
            cache=cache, progress=progress,
        )
        for fa, fb in file_pairs
    ]
    results = await asyncio.gather(*tasks)

    non_empty = [r for r in results if r.similar_functions]
    logger.info(
        "Comparison done: %d pairs had similar functions out of %d total "
        "(cached=%d, new=%d, skipped=%d, errors=%d)",
        len(non_empty), len(results),
        progress["cached"], progress["completed"],
        progress["skipped"], progress["errors"],
    )
    return list(results)


def deduplicate_results(results: list[CompareResult]) -> list[CompareResult]:
    """Deduplicate similar function findings across all compare results.

    The same function pair might be found from different file comparisons.
    Keep the entry with the highest similarity score.
    """
    best: dict[tuple[str, str, str, str], SimilarFunction] = {}
    for cr in results:
        for sf in cr.similar_functions:
            key = (sf.func_a.file, sf.func_a.name, sf.func_b.file, sf.func_b.name)
            if key not in best or sf.composite_score > best[key].composite_score:
                best[key] = sf

    deduped: dict[tuple[str, str], CompareResult] = {}
    for (fa_file, _, fb_file, _), sf in best.items():
        pair_key = (fa_file, fb_file)
        if pair_key not in deduped:
            deduped[pair_key] = CompareResult(file_a=fa_file, file_b=fb_file)
        deduped[pair_key].similar_functions.append(sf)

    return [cr for cr in deduped.values() if cr.similar_functions]
