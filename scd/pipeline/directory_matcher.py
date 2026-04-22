"""Directory matching based on pre-generated summaries."""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher

from scd.ai.client import ClaudeClient
from scd.ai.prompts import DIRECTORY_MATCH_SYSTEM, DIRECTORY_MATCH_USER
from scd.models import DirMatch, DirMatchResult, RepoScanResult

logger = logging.getLogger(__name__)
DIRECTORY_MATCH_MAX_TOKENS = 1200
HEURISTIC_MIN_SCORE = 0.22


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
    """Fallback matching when remote model is unavailable."""
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


async def match_directories(
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
    summaries_a: dict[str, str],
    summaries_b: dict[str, str],
    client: ClaudeClient,
) -> DirMatchResult:
    """Match directories between two repos using their pre-generated summaries."""
    logger.info(
        "Matching directories via summaries (%d dirs in A, %d dirs in B)",
        len(summaries_a), len(summaries_b),
    )

    user_msg = DIRECTORY_MATCH_USER.format(
        repo_a_summaries=_format_summaries(summaries_a),
        repo_b_summaries=_format_summaries(summaries_b),
    )
    logger.info(
        "Directory match prompt size: %d chars, max_tokens=%d",
        len(user_msg),
        DIRECTORY_MATCH_MAX_TOKENS,
    )

    try:
        data = await client.ask_json(
            system=DIRECTORY_MATCH_SYSTEM,
            user=user_msg,
            max_tokens=DIRECTORY_MATCH_MAX_TOKENS,
        )
        result = DirMatchResult()
    except Exception as e:
        logger.error("Directory matching via model failed, fallback to heuristic: %s", e)
        result = _heuristic_match(repo_a, repo_b, summaries_a, summaries_b)
        logger.info("Heuristic directory matching done: %d pairs", len(result.matched_dirs))
        return result

    for m in data.get("matched_dirs", []):
        dir_a = m.get("dir_a", "")
        dir_b = m.get("dir_b", "")
        if dir_a in repo_a.dirs and dir_b in repo_b.dirs:
            result.matched_dirs.append(DirMatch(
                dir_a=dir_a,
                dir_b=dir_b,
                confidence=m.get("confidence", "medium"),
                reason=m.get("reason", ""),
            ))
        else:
            logger.warning("AI returned invalid dir pair: %s <-> %s", dir_a, dir_b)

    logger.info("Directory matching done: %d pairs", len(result.matched_dirs))
    return result
