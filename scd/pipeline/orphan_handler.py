from __future__ import annotations

import logging

from scd.ai.client import ClaudeClient
from scd.ai.prompts import ORPHAN_CHECK_SYSTEM, ORPHAN_CHECK_USER
from scd.models import DirInfo, DirMatch, RepoScanResult

logger = logging.getLogger(__name__)


def _format_dir_structure(repo: RepoScanResult) -> str:
    """Build a compact text summary of a repo's directory tree with file listings."""
    lines: list[str] = []
    for dir_path in sorted(repo.dirs.keys()):
        dir_info: DirInfo = repo.dirs[dir_path]
        label = dir_path or "(root)"
        file_names = [f.path.rsplit("/", 1)[-1] for f in dir_info.files]
        langs = dir_info.lang_distribution
        lang_str = ", ".join(f"{k}:{v}" for k, v in sorted(langs.items()))
        lines.append(f"{label}/  [{lang_str}]  files: {', '.join(file_names)}")
    return "\n".join(lines)


def _format_orphan_list(orphan_dirs: list[str], repo: RepoScanResult) -> str:
    """Format orphan directory list with their file details."""
    if not orphan_dirs:
        return "(none)"
    parts: list[str] = []
    for d in orphan_dirs:
        dir_info = repo.dirs.get(d)
        if dir_info and dir_info.files:
            file_names = [f.path.rsplit("/", 1)[-1] for f in dir_info.files]
            parts.append(f"  {d or '(root)'}: {', '.join(file_names)}")
        else:
            parts.append(f"  {d or '(root)'}: (no source files)")
    return "\n".join(parts)


async def handle_orphan_dirs(
    orphan_dirs_a: list[str],
    orphan_dirs_b: list[str],
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
    client: ClaudeClient,
) -> list[DirMatch]:
    """Check all orphan directories for potential matches in a single AI call."""
    if not orphan_dirs_a and not orphan_dirs_b:
        return []

    user_msg = ORPHAN_CHECK_USER.format(
        orphan_dirs_a=_format_orphan_list(orphan_dirs_a, repo_a),
        orphan_dirs_b=_format_orphan_list(orphan_dirs_b, repo_b),
        repo_a_structure=_format_dir_structure(repo_a),
        repo_b_structure=_format_dir_structure(repo_b),
    )

    try:
        data = await client.ask_json(system=ORPHAN_CHECK_SYSTEM, user=user_msg)
    except Exception as e:
        logger.error("Error checking orphan dirs: %s", e)
        return []

    matches: list[DirMatch] = []
    for pm in data.get("potential_matches", []):
        orphan_dir = pm.get("orphan_dir", "")
        candidate = pm.get("candidate_dir", "")
        orphan_repo = pm.get("orphan_repo", "").upper()

        if orphan_repo == "A" and candidate in repo_b.dirs:
            matches.append(DirMatch(
                dir_a=orphan_dir, dir_b=candidate,
                confidence="low", reason=pm.get("reason", "orphan recovery"),
            ))
        elif orphan_repo == "B" and candidate in repo_a.dirs:
            matches.append(DirMatch(
                dir_a=candidate, dir_b=orphan_dir,
                confidence="low", reason=pm.get("reason", "orphan recovery"),
            ))

    logger.info("Orphan handling recovered %d additional directory pairs", len(matches))
    return matches
