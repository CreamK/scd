from __future__ import annotations

import asyncio
import logging
from itertools import product

from scd.ai.client import ClaudeClient
from scd.ai.prompts import FUNCTION_COMPARE_SYSTEM, FUNCTION_COMPARE_USER
from scd.config import ScdConfig
from scd.models import (
    CompareResult,
    DirMatch,
    FuncLocation,
    RepoScanResult,
    SimilarFunction,
    SimilarityType,
)

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
            sim_type_str = item.get("similarity_type", "partial")
            try:
                sim_type = SimilarityType(sim_type_str)
            except ValueError:
                sim_type = SimilarityType.PARTIAL

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
                similarity_score=int(item.get("similarity_score", 0)),
                similarity_type=sim_type,
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
) -> CompareResult:
    """Compare two files using Claude AI."""
    code_a = repo_a.file_contents.get(file_a, "")
    code_b = repo_b.file_contents.get(file_b, "")

    if not code_a or not code_b:
        return CompareResult(file_a=file_a, file_b=file_b)

    system = FUNCTION_COMPARE_SYSTEM.format(threshold=threshold)
    user = FUNCTION_COMPARE_USER.format(
        file_a=file_a, file_b=file_b,
        code_a=code_a, code_b=code_b,
    )

    try:
        data = await client.ask_json(system, user)
        similar = _parse_similar_functions(data, file_a, file_b)
        return CompareResult(file_a=file_a, file_b=file_b, similar_functions=similar)
    except Exception as e:
        logger.error("Error comparing %s <-> %s: %s", file_a, file_b, e)
        return CompareResult(file_a=file_a, file_b=file_b)


async def compare_matched_dirs(
    matched_dirs: list[DirMatch],
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
    client: ClaudeClient,
    config: ScdConfig,
) -> list[CompareResult]:
    """Compare all file pairs across all matched directory pairs."""
    all_pairs: list[tuple[str, str]] = []
    for match in matched_dirs:
        pairs = _build_file_pairs(match, repo_a, repo_b)
        all_pairs.extend(pairs)
        logger.info(
            "Dir pair %s <-> %s: %d file pairs",
            match.dir_a, match.dir_b, len(pairs),
        )

    logger.info("Total file pairs to compare: %d", len(all_pairs))

    tasks = [
        _compare_file_pair(fa, fb, repo_a, repo_b, client, config.similarity_threshold)
        for fa, fb in all_pairs
    ]
    results = await asyncio.gather(*tasks)

    non_empty = [r for r in results if r.similar_functions]
    logger.info(
        "Comparison done: %d pairs had similar functions out of %d total",
        len(non_empty), len(results),
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
            if key not in best or sf.similarity_score > best[key].similarity_score:
                best[key] = sf

    deduped: dict[tuple[str, str], CompareResult] = {}
    for (fa_file, _, fb_file, _), sf in best.items():
        pair_key = (fa_file, fb_file)
        if pair_key not in deduped:
            deduped[pair_key] = CompareResult(file_a=fa_file, file_b=fb_file)
        deduped[pair_key].similar_functions.append(sf)

    return [cr for cr in deduped.values() if cr.similar_functions]
