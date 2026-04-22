from __future__ import annotations

import asyncio
import logging
from itertools import product

from scd.ai.client import ClaudeClient
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
from scd.pipeline.compare_cache import PairCache, compute_pair_key

logger = logging.getLogger(__name__)


def _build_file_pairs(
    match: DirMatch,
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
) -> list[tuple[str, str]]:
    """Generate all file pairs (n x m) for a matched directory pair."""
    dir_a = repo_a.dirs.get(match.dir_a)
    dir_b = repo_b.dirs.get(match.dir_b)
    if not dir_a or not dir_b:
        return []

    files_a = [f.path for f in dir_a.files]
    files_b = [f.path for f in dir_b.files]
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
    client: ClaudeClient,
    threshold: int,
    model: str,
    cache: PairCache | None = None,
    progress: dict | None = None,
) -> CompareResult:
    """Compare two files using Claude AI, with optional checkpoint cache."""
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
    """Build all file pairs (n x m) from matched directory pairs."""
    all_pairs: list[tuple[str, str]] = []
    for match in matched_dirs:
        pairs = _build_file_pairs(match, repo_a, repo_b)
        all_pairs.extend(pairs)
        logger.info(
            "Dir pair %s <-> %s: %d file pairs",
            match.dir_a, match.dir_b, len(pairs),
        )
    logger.info("Total file pairs to compare: %d", len(all_pairs))
    return all_pairs


async def compare_file_pairs(
    file_pairs: list[tuple[str, str]],
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
    client: ClaudeClient,
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
