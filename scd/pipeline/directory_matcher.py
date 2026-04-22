"""Directory matching based on pre-generated summaries."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from difflib import SequenceMatcher

from scd.ai.client import ClaudeClient
from scd.ai.prompts import DIRECTORY_MATCH_SYSTEM, DIRECTORY_MATCH_USER
from scd.models import DirMatch, DirMatchResult, RepoScanResult

logger = logging.getLogger(__name__)
DIRECTORY_MATCH_MAX_TOKENS = 3072
HEURISTIC_MIN_SCORE = 0.22
_CONF_ORDER = {"high": 3, "medium": 2, "low": 1}


def _normalize_ai_dir_path(raw: str) -> str:
    """Map model output to `repo.dirs` keys. Root is stored as \"\"; the prompt shows [(root)]."""
    s = (raw or "").strip()
    if s == "(root)":
        return ""
    return s


def _summary_to_text(summary: str) -> str:
    """Convert JSON summary into plain text for heuristic matching."""
    try:
        data = json.loads(summary)
        fields = [
            data.get("purpose", ""),
            " ".join(data.get("key_exports", [])),
            " ".join(data.get("frameworks", [])),
            " ".join(data.get("patterns", [])),
            data.get("children_overview", ""),
        ]
        return " ".join(part for part in fields if part)
    except (json.JSONDecodeError, TypeError, ValueError):
        return summary


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return {t for t in tokens if len(t) >= 3}


def _dir_similarity(dir_a: str, sum_a: str, dir_b: str, sum_b: str) -> float:
    """Combine path-name similarity and summary-token overlap."""
    name_score = SequenceMatcher(None, dir_a or "root", dir_b or "root").ratio()
    toks_a = _tokenize(_summary_to_text(sum_a))
    toks_b = _tokenize(_summary_to_text(sum_b))
    if toks_a and toks_b:
        overlap = len(toks_a & toks_b) / len(toks_a | toks_b)
    else:
        overlap = 0.0
    return name_score * 0.35 + overlap * 0.65


def _heuristic_match(
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
    summaries_a: dict[str, str],
    summaries_b: dict[str, str],
) -> DirMatchResult:
    """Fallback matching when the model is unavailable."""
    candidates: list[tuple[float, str, str]] = []
    for dir_a, sum_a in summaries_a.items():
        for dir_b, sum_b in summaries_b.items():
            score = _dir_similarity(dir_a, sum_a, dir_b, sum_b)
            if score >= HEURISTIC_MIN_SCORE:
                candidates.append((score, dir_a, dir_b))
    candidates.sort(reverse=True, key=lambda x: x[0])

    used_a: set[str] = set()
    used_b: set[str] = set()
    result = DirMatchResult()
    for score, dir_a, dir_b in candidates:
        if dir_a in used_a or dir_b in used_b:
            continue
        if dir_a not in repo_a.dirs or dir_b not in repo_b.dirs:
            continue
        confidence = "high" if score >= 0.55 else "medium" if score >= 0.40 else "low"
        result.matched_dirs.append(
            DirMatch(
                dir_a=dir_a,
                dir_b=dir_b,
                confidence=confidence,
                reason=f"heuristic fallback score={score:.2f}",
            ),
        )
        used_a.add(dir_a)
        used_b.add(dir_b)
    return result


def _format_summaries(summaries: dict[str, str]) -> str:
    """Format directory summaries for the matching prompt."""
    lines: list[str] = []
    for dir_path in sorted(summaries.keys()):
        label = dir_path or "(root)"
        lines.append(f"[{label}]: {summaries[dir_path]}")
    return "\n".join(lines)


def _chunk(summaries: dict[str, str], size: int) -> list[dict[str, str]]:
    """Split summaries into deterministic chunks of at most `size` entries."""
    items = sorted(summaries.items())
    if size <= 0:
        return [dict(items)] if items else []
    return [dict(items[i:i + size]) for i in range(0, len(items), size)]


def _resolve_one_to_one(matches: list[DirMatch]) -> DirMatchResult:
    """Collapse cross-batch duplicates into a one-to-one mapping.

    Higher-confidence matches win; ties are resolved first-seen.
    """
    matches_sorted = sorted(
        matches,
        key=lambda m: _CONF_ORDER.get(m.confidence, 0),
        reverse=True,
    )
    used_a: set[str] = set()
    used_b: set[str] = set()
    kept: list[DirMatch] = []
    for m in matches_sorted:
        if m.dir_a in used_a or m.dir_b in used_b:
            continue
        kept.append(m)
        used_a.add(m.dir_a)
        used_b.add(m.dir_b)
    return DirMatchResult(matched_dirs=kept)


async def _match_single_batch(
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
    summaries_a: dict[str, str],
    summaries_b: dict[str, str],
    client: ClaudeClient,
) -> DirMatchResult:
    """Run one AI call over a pair of summary batches."""
    user_msg = DIRECTORY_MATCH_USER.format(
        repo_a_summaries=_format_summaries(summaries_a),
        repo_b_summaries=_format_summaries(summaries_b),
    )

    data = await client.ask_json(
        system=DIRECTORY_MATCH_SYSTEM,
        user=user_msg,
        max_tokens=DIRECTORY_MATCH_MAX_TOKENS,
    )

    result = DirMatchResult()
    for m in data.get("matched_dirs", []):
        dir_a = _normalize_ai_dir_path(m.get("dir_a", ""))
        dir_b = _normalize_ai_dir_path(m.get("dir_b", ""))
        if dir_a in repo_a.dirs and dir_b in repo_b.dirs:
            result.matched_dirs.append(DirMatch(
                dir_a=dir_a,
                dir_b=dir_b,
                confidence=m.get("confidence", "medium"),
                reason=m.get("reason", ""),
            ))
        else:
            logger.warning("AI returned invalid dir pair: %s <-> %s", dir_a, dir_b)
    return result


async def _match_in_batches(
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
    summaries_a: dict[str, str],
    summaries_b: dict[str, str],
    client: ClaudeClient,
    batch_size: int,
) -> DirMatchResult:
    """Run p*q concurrent batch calls and merge results into a one-to-one mapping."""
    a_batches = _chunk(summaries_a, batch_size)
    b_batches = _chunk(summaries_b, batch_size)
    total_calls = len(a_batches) * len(b_batches)
    logger.info(
        "Matching in batches: %d x %d = %d AI calls (batch_size=%d)",
        len(a_batches), len(b_batches), total_calls, batch_size,
    )

    tasks = [
        _match_single_batch(repo_a, repo_b, a_batch, b_batch, client)
        for a_batch in a_batches
        for b_batch in b_batches
    ]
    sub_results = await asyncio.gather(*tasks, return_exceptions=True)

    all_matches: list[DirMatch] = []
    failed = 0
    for r in sub_results:
        if isinstance(r, Exception):
            logger.warning("Sub-batch failed, skipping: %s", r)
            failed += 1
            continue
        all_matches.extend(r.matched_dirs)

    if failed:
        logger.warning("%d/%d sub-batches failed", failed, total_calls)
    if failed == total_calls:
        raise RuntimeError("All directory-match sub-batches failed")

    return _resolve_one_to_one(all_matches)


async def match_directories(
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
    summaries_a: dict[str, str],
    summaries_b: dict[str, str],
    client: ClaudeClient,
    batch_size: int = 40,
) -> DirMatchResult:
    """Match directories between two repos using their pre-generated summaries.

    Falls back to heuristic matching if the model-based path raises.
    """
    n, m = len(summaries_a), len(summaries_b)
    logger.info(
        "Matching directories via summaries (%d dirs in A, %d dirs in B)",
        n, m,
    )
    if n == 0 or m == 0:
        return DirMatchResult()

    try:
        if n <= batch_size and m <= batch_size:
            result = await _match_single_batch(
                repo_a, repo_b, summaries_a, summaries_b, client,
            )
        else:
            result = await _match_in_batches(
                repo_a, repo_b, summaries_a, summaries_b, client, batch_size,
            )
    except Exception as e:
        logger.error("Directory matching via model failed, fallback to heuristic: %s", e)
        result = _heuristic_match(repo_a, repo_b, summaries_a, summaries_b)
        logger.info("Heuristic directory matching done: %d pairs", len(result.matched_dirs))
        return result

    logger.info("Directory matching done: %d pairs", len(result.matched_dirs))
    return result
