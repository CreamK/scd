from __future__ import annotations

import asyncio
import logging
import os

from scd.ai.client import ClaudeClient
from scd.ai.prompts import ORPHAN_CHECK_SYSTEM, ORPHAN_CHECK_USER
from scd.ai.tools import ALL_TOOLS, create_tool_handler
from scd.models import DirMatch, RepoScanResult

logger = logging.getLogger(__name__)


async def _check_orphan_dir(
    orphan_dir: str,
    repo_label: str,
    source_repo: RepoScanResult,
    other_repo: RepoScanResult,
    client: ClaudeClient,
) -> list[DirMatch]:
    """Check if an orphan directory has matches in the other repo via agent exploration."""
    dir_info = source_repo.dirs.get(orphan_dir)
    if not dir_info or not dir_info.files:
        return []

    orphan_dir_abs = os.path.join(source_repo.root_path, orphan_dir) if orphan_dir else source_repo.root_path
    tool_handler = create_tool_handler([source_repo.root_path, other_repo.root_path])

    user_msg = ORPHAN_CHECK_USER.format(
        orphan_dir=orphan_dir,
        orphan_dir_abs=orphan_dir_abs,
        repo_label=repo_label,
        other_repo_path=other_repo.root_path,
    )

    try:
        data = await client.agent_loop_json(
            system=ORPHAN_CHECK_SYSTEM,
            user=user_msg,
            tools=ALL_TOOLS,
            tool_handler=tool_handler,
            max_turns=20,
        )
    except Exception as e:
        logger.error("Error checking orphan dir %s: %s", orphan_dir, e)
        return []

    matches: list[DirMatch] = []
    for pm in data.get("potential_matches", []):
        candidate = pm.get("candidate_dir", "")
        if candidate in other_repo.dirs:
            if repo_label == "A":
                matches.append(DirMatch(
                    dir_a=orphan_dir, dir_b=candidate,
                    confidence="low", reason=pm.get("reason", "orphan recovery"),
                ))
            else:
                matches.append(DirMatch(
                    dir_a=candidate, dir_b=orphan_dir,
                    confidence="low", reason=pm.get("reason", "orphan recovery"),
                ))
    return matches


async def handle_orphan_dirs(
    orphan_dirs_a: list[str],
    orphan_dirs_b: list[str],
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
    client: ClaudeClient,
) -> list[DirMatch]:
    """Check all orphan directories for potential matches."""
    tasks = []
    for d in orphan_dirs_a:
        tasks.append(_check_orphan_dir(d, "A", repo_a, repo_b, client))
    for d in orphan_dirs_b:
        tasks.append(_check_orphan_dir(d, "B", repo_b, repo_a, client))

    results = await asyncio.gather(*tasks)

    all_matches: list[DirMatch] = []
    for match_list in results:
        all_matches.extend(match_list)

    logger.info("Orphan handling recovered %d additional directory pairs", len(all_matches))
    return all_matches
