from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import NamedTuple

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
from scd.pipeline.dir_summarizer import _count_tokens

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
PAIR_CACHE_VERSION = 4
STRICT_COMPOSITE_MIN_SCORE = 80
DATA_STRUCTURE_MIN_SCORE = 75
ALGORITHM_LOGIC_MIN_SCORE = 80
NAMING_CONVENTION_MIN_SCORE = 50

# Token budgeting for a single Phase 3 request. The per-pair input budget is
# derived from the model context window minus the reserved response tokens and
# the (system + user template) prompt overhead, scaled by a safety factor
# because the token estimate is heuristic.
RESPONSE_TOKENS_RESERVED = 8192  # matches LlmClient.ask_json default max_tokens
PROMPT_OVERHEAD_TOKENS = 2000
TOKEN_BUDGET_SAFETY = 0.9
# Estimated tokens added per line by the "NNNNN | " prefix and the newline.
LINE_OVERHEAD_TOKENS = 4


def compute_pair_key(
    file_a: str,
    content_a: str,
    file_b: str,
    content_b: str,
    model: str,
    threshold: int,
    span_a: tuple[int, int] | None = None,
    span_b: tuple[int, int] | None = None,
) -> str:
    """Stable key for a (file_a, file_b) comparison under given model/threshold.

    ``content_a``/``content_b`` are the exact code sent to the model (the whole
    file, or one chunk of it). ``span_a``/``span_b`` are the 1-based inclusive
    line ranges of those chunks within their files, so different chunks of the
    same file get distinct keys.
    """
    h = hashlib.sha256()
    h.update(f"v{PAIR_CACHE_VERSION}".encode())
    h.update(b"\0")
    h.update(model.encode())
    h.update(b"\0")
    h.update(str(threshold).encode())
    h.update(b"\0")
    h.update(f"{span_a}|{span_b}".encode())
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
                    if data.get("v") != PAIR_CACHE_VERSION:
                        continue
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


def _annotate_with_line_numbers(code: str, start_line: int = 1) -> str:
    """Prefix every line with its 1-based line number for LLM consumption.

    The resulting format is ``"%Nd | <line>"`` where N auto-scales to the file
    length (minimum 5). Giving the model authoritative line numbers up front
    avoids the well-known failure mode of LLMs miscounting lines in long files,
    which is the dominant source of bogus ``line_start``/``line_end`` values in
    similar-function output.

    ``start_line`` is the file line number of the first line in ``code``, so a
    chunk taken from the middle of a file keeps its original line numbers.
    """
    if not code:
        return code
    lines = code.splitlines()
    width = max(5, len(str(start_line + len(lines) - 1)))
    return "\n".join(
        f"{i:>{width}} | {line}" for i, line in enumerate(lines, start_line)
    )


# ---------------------------------------------------------------------------
# Token budgeting & chunking
#
# A single Phase 3 request must fit the model context window. When the two
# files together exceed the input budget, the oversized side(s) are split into
# chunks along function boundaries and every chunk-a x chunk-b combination is
# compared separately. Chunks keep their original file line numbers, so the
# rest of the pipeline (parsing, dedup, reporting) is unaffected.
# ---------------------------------------------------------------------------


class CodeChunk(NamedTuple):
    """A contiguous slice of a file: 1-based inclusive line range plus text."""

    start_line: int
    end_line: int
    code: str


def _estimate_code_tokens(code: str) -> int:
    """Token estimate for code as it will be sent (with line-number prefixes).

    Uses the same tiktoken counter as Phase 2 (cl100k_base, vendored offline
    cache); TOKEN_BUDGET_SAFETY absorbs cross-tokenizer skew vs the deployed
    model. LINE_OVERHEAD_TOKENS covers the "NNNNN | " prefix added later by
    ``_annotate_with_line_numbers``.
    """
    line_count = code.count("\n") + 1 if code else 0
    return _count_tokens(code) + LINE_OVERHEAD_TOKENS * line_count


def pair_token_budget(context_window: int) -> int:
    """Input token budget for both code blocks of one comparison request."""
    budget = int(
        (context_window - RESPONSE_TOKENS_RESERVED - PROMPT_OVERHEAD_TOKENS)
        * TOKEN_BUDGET_SAFETY
    )
    return max(budget, 2000)


# Column-0 lines that plausibly start a new top-level definition. Used only to
# pick chunk split points, so false negatives are fine and false positives are
# cheap (a slightly suboptimal split).
_TOP_LEVEL_DEF_RE = re.compile(
    r"^(?:async\s+def\s|def\s|class\s|func\s|fn\s|function\s|"
    r"impl[\s<]|struct\s|enum\s|union\s|interface\s|trait\s|"
    r"(?:public|private|protected|internal|export|static|abstract|final|"
    r"const|unsafe|inline|extern|virtual|override)\s|"
    r"[A-Za-z_][\w:<>,.\s*&\[\]]*\s[*&]*[A-Za-z_]\w*\s*\()"
)


def _boundary_strength(lines: list[str], i: int) -> int:
    """How good is line index ``i`` (0-based) as the first line of a new chunk?

    2 = strong (top-level definition start, or right after a column-0 brace)
    1 = weak (right after a blank line)
    0 = not a boundary
    """
    if i <= 0 or i >= len(lines):
        return 0
    line = lines[i]
    if line[:1].isspace() or not line:
        prev = lines[i - 1].strip()
        return 1 if not prev else 0
    if _TOP_LEVEL_DEF_RE.match(line):
        return 2
    prev = lines[i - 1].strip()
    if prev in {"}", "};", "});", ")", ");"}:
        return 2
    if not prev:
        return 1
    return 0


def split_code_into_chunks(code: str, max_tokens: int) -> list[CodeChunk]:
    """Split code into chunks of at most ~max_tokens, preferring function boundaries.

    Greedy: extend the current chunk as far as the budget allows, then cut at
    the best boundary inside the window (strong > weak > hard cut). Line
    numbers in the returned chunks are 1-based and inclusive.
    """
    lines = code.splitlines()
    if not lines:
        return [CodeChunk(1, 1, code)]

    costs = [
        _count_tokens(line) + LINE_OVERHEAD_TOKENS for line in lines
    ]

    chunks: list[CodeChunk] = []
    start = 0
    while start < len(lines):
        acc = 0
        end = start
        while end < len(lines) and acc + costs[end] <= max_tokens:
            acc += costs[end]
            end += 1
        if end >= len(lines):
            chunks.append(_make_chunk(lines, start, len(lines)))
            break
        if end == start:
            # Single line exceeds the budget; emit it alone rather than loop.
            end = start + 1

        cut = end
        best_strength = 0
        for i in range(end, start, -1):
            strength = _boundary_strength(lines, i)
            if strength > best_strength:
                best_strength = strength
                cut = i
                if strength == 2:
                    break
        chunks.append(_make_chunk(lines, start, cut))
        start = cut

    return chunks


def _make_chunk(lines: list[str], start: int, end: int) -> CodeChunk:
    """Build a CodeChunk from 0-based [start, end) line indices."""
    return CodeChunk(start + 1, end, "\n".join(lines[start:end]))


def _plan_pair_chunks(
    code_a: str,
    code_b: str,
    budget: int,
) -> tuple[list[CodeChunk], list[CodeChunk]]:
    """Decide how to chunk each side so any chunk-a + chunk-b fits the budget.

    Only oversized sides are split; when one side is small enough it stays
    whole and the other side gets all remaining budget (m x 1 instead of m x n).
    """
    def _whole(code: str) -> CodeChunk:
        return CodeChunk(1, code.count("\n") + 1, code)

    est_a = _estimate_code_tokens(code_a)
    est_b = _estimate_code_tokens(code_b)
    if est_a + est_b <= budget:
        return [_whole(code_a)], [_whole(code_b)]

    half = budget // 2
    if est_b <= half:
        return split_code_into_chunks(code_a, budget - est_b), [_whole(code_b)]
    if est_a <= half:
        return [_whole(code_a)], split_code_into_chunks(code_b, budget - est_a)
    return (
        split_code_into_chunks(code_a, half),
        split_code_into_chunks(code_b, half),
    )


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
    files in nested subdirectories also participate in comparison. Only files
    with the same extension are paired; cross-match duplicates are removed
    later by ``build_all_file_pairs``.
    """
    if match.dir_a not in repo_a.dirs or match.dir_b not in repo_b.dirs:
        return []

    files_a = _iter_subtree_files(repo_a, match.dir_a)
    files_b = _iter_subtree_files(repo_b, match.dir_b)
    return [
        (file_a, file_b)
        for file_a, file_b in product(files_a, files_b)
        if Path(file_a).suffix.lower() == Path(file_b).suffix.lower()
    ]


def _composite_from_scores(scores: DimensionScores) -> int:
    """Recompute the weighted composite from the per-dimension scores.

    The model's self-reported ``composite_score`` is intentionally ignored: we
    always derive the composite here so the documented weighting is actually
    enforced and cannot drift from what the model claims.
    """
    weighted = (
        scores.data_structure * 0.40
        + scores.function_signature * 0.10
        + scores.algorithm_logic * 0.40
        + scores.naming_convention * 0.05
        + scores.protocol_conformance * 0.05
    )
    return round(weighted)


def _level_from_composite(composite: int) -> SimilarityLevel:
    """Map a composite score to a similarity level (matches the prompt bands)."""
    if composite > 60:
        return SimilarityLevel.HIGH
    if composite >= 40:
        return SimilarityLevel.MEDIUM
    if composite >= 20:
        return SimilarityLevel.LOW
    return SimilarityLevel.VERY_LOW


def _passes_implementation_similarity_gate(
    scores: DimensionScores,
    composite_score: int,
    threshold: int,
) -> bool:
    """Require visible implementation overlap, not only functional equivalence."""
    if composite_score < max(threshold, STRICT_COMPOSITE_MIN_SCORE):
        return False

    return (
        scores.data_structure >= DATA_STRUCTURE_MIN_SCORE
        and scores.algorithm_logic >= ALGORITHM_LOGIC_MIN_SCORE
        and scores.naming_convention >= NAMING_CONVENTION_MIN_SCORE
    )


def _reported_file_matches(reported: object, expected: str) -> bool:
    """Return False only when the model explicitly reports the wrong file."""
    if reported in (None, ""):
        return True
    return str(reported).strip() == expected


def _parse_similar_functions(
    data: dict,
    file_a: str,
    file_b: str,
    threshold: int = 20,
) -> list[SimilarFunction]:
    """Parse AI response into SimilarFunction objects."""
    results: list[SimilarFunction] = []
    for item in data.get("similar_functions", []):
        try:
            fa = item["func_a"]
            fb = item["func_b"]

            if (
                not _reported_file_matches(fa.get("file"), file_a)
                or not _reported_file_matches(fb.get("file"), file_b)
            ):
                logger.debug(
                    "Filtered non-cross-file function pair: func_a.file=%r "
                    "(expected %r), func_b.file=%r (expected %r)",
                    fa.get("file"),
                    file_a,
                    fb.get("file"),
                    file_b,
                )
                continue

            raw_scores = item.get("scores", {})
            scores = DimensionScores(
                data_structure=int(raw_scores.get("data_structure", 0)),
                function_signature=int(raw_scores.get("function_signature", 0)),
                algorithm_logic=int(raw_scores.get("algorithm_logic", 0)),
                naming_convention=int(raw_scores.get("naming_convention", 0)),
                protocol_conformance=int(raw_scores.get("protocol_conformance", 0)),
            )

            composite = _composite_from_scores(scores)
            level = _level_from_composite(composite)

            similar = SimilarFunction(
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
            )
            if _passes_implementation_similarity_gate(scores, composite, threshold):
                results.append(similar)
            else:
                logger.debug(
                    "Filtered function pair %s <-> %s: composite=%d, "
                    "data_structure=%d, algorithm_logic=%d, naming_convention=%d",
                    similar.func_a.name,
                    similar.func_b.name,
                    composite,
                    scores.data_structure,
                    scores.algorithm_logic,
                    scores.naming_convention,
                )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Failed to parse similar function entry: %s", e)
    return results


async def _compare_chunk_pair(
    file_a: str,
    chunk_a: CodeChunk,
    file_b: str,
    chunk_b: CodeChunk,
    client: LlmClient,
    effective_threshold: int,
    model: str,
    cache: PairCache | None,
) -> tuple[list[SimilarFunction] | None, bool]:
    """Compare one chunk pair. Returns (similar_functions, was_cached).

    ``similar_functions`` is None when the LLM call failed.
    """
    span_a = (chunk_a.start_line, chunk_a.end_line)
    span_b = (chunk_b.start_line, chunk_b.end_line)

    cache_key: str | None = None
    if cache is not None:
        cache_key = compute_pair_key(
            file_a, chunk_a.code, file_b, chunk_b.code,
            model, effective_threshold, span_a, span_b,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached.similar_functions, True

    system = FUNCTION_COMPARE_SYSTEM.format(
        threshold=effective_threshold,
        file_a=file_a,
        file_b=file_b,
    )
    user = FUNCTION_COMPARE_USER.format(
        file_a=file_a, file_b=file_b,
        code_a=_annotate_with_line_numbers(chunk_a.code, chunk_a.start_line),
        code_b=_annotate_with_line_numbers(chunk_b.code, chunk_b.start_line),
    )

    try:
        data = await client.ask_json(system, user)
    except Exception as e:
        logger.error(
            "Error comparing %s [%d-%d] <-> %s [%d-%d]: %s",
            file_a, *span_a, file_b, *span_b, e,
        )
        return None, False

    similar = _parse_similar_functions(data, file_a, file_b, effective_threshold)
    if cache is not None and cache_key is not None:
        await cache.put(
            cache_key,
            CompareResult(file_a=file_a, file_b=file_b, similar_functions=similar),
        )
    return similar, False


def _merge_chunk_findings(findings: list[SimilarFunction]) -> list[SimilarFunction]:
    """Deduplicate findings across chunk pairs, keeping the highest score."""
    best: dict[tuple[str, int, str, int], SimilarFunction] = {}
    for sf in findings:
        key = (sf.func_a.name, sf.func_a.line_start, sf.func_b.name, sf.func_b.line_start)
        if key not in best or sf.composite_score > best[key].composite_score:
            best[key] = sf
    return list(best.values())


async def _compare_file_pair(
    file_a: str,
    file_b: str,
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
    client: LlmClient,
    threshold: int,
    model: str,
    context_window: int = 128_000,
    cache: PairCache | None = None,
    progress: dict | None = None,
) -> CompareResult:
    """Compare two files using the configured LLM, with optional checkpoint cache.

    When the pair does not fit the per-request token budget, each oversized
    side is split into chunks along function boundaries and all chunk
    combinations are compared; findings are merged back into one result.
    """
    code_a = repo_a.file_contents.get(file_a, "")
    code_b = repo_b.file_contents.get(file_b, "")

    if not code_a or not code_b:
        if progress is not None:
            progress["skipped"] = progress.get("skipped", 0) + 1
        return CompareResult(file_a=file_a, file_b=file_b)

    effective_threshold = max(threshold, STRICT_COMPOSITE_MIN_SCORE)
    budget = pair_token_budget(context_window)
    chunks_a, chunks_b = _plan_pair_chunks(code_a, code_b, budget)
    sub_pairs = list(product(chunks_a, chunks_b))
    if len(sub_pairs) > 1:
        logger.info(
            "Pair %s <-> %s exceeds the %d-token budget; comparing as %dx%d "
            "chunk pairs",
            file_a, file_b, budget, len(chunks_a), len(chunks_b),
        )

    outcomes = await asyncio.gather(*(
        _compare_chunk_pair(
            file_a, ca, file_b, cb, client, effective_threshold, model, cache,
        )
        for ca, cb in sub_pairs
    ))

    findings: list[SimilarFunction] = []
    had_error = False
    all_cached = True
    for similar, was_cached in outcomes:
        if similar is None:
            had_error = True
            all_cached = False
            continue
        if not was_cached:
            all_cached = False
        findings.extend(similar)

    result = CompareResult(
        file_a=file_a,
        file_b=file_b,
        similar_functions=_merge_chunk_findings(findings),
    )

    if progress is not None:
        if had_error:
            progress["errors"] = progress.get("errors", 0) + 1
        elif all_cached:
            progress["cached"] = progress.get("cached", 0) + 1
        else:
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
    on_similar_result: Callable[[CompareResult], None] | None = None,
) -> list[CompareResult]:
    """Compare file pairs, optionally streaming non-empty results as they finish."""
    progress: dict = {
        "total": len(file_pairs),
        "cached": 0,
        "completed": 0,
        "skipped": 0,
        "errors": 0,
        "log_every": 25,
    }

    async def run_one(index: int, file_a: str, file_b: str) -> tuple[int, CompareResult]:
        result = await _compare_file_pair(
            file_a, file_b, repo_a, repo_b, client,
            config.similarity_threshold, config.model,
            context_window=config.context_window,
            cache=cache, progress=progress,
        )
        return index, result

    tasks = [
        asyncio.create_task(run_one(i, fa, fb))
        for i, (fa, fb) in enumerate(file_pairs)
    ]
    results_by_index: list[CompareResult | None] = [None] * len(tasks)

    for task in asyncio.as_completed(tasks):
        index, result = await task
        results_by_index[index] = result
        if on_similar_result is not None and result.similar_functions:
            on_similar_result(result)

    results = [r for r in results_by_index if r is not None]

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
