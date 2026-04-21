from __future__ import annotations

import logging

from scd.ai.client import ClaudeClient
from scd.ai.prompts import DIRECTORY_MATCH_SYSTEM, DIRECTORY_MATCH_USER
from scd.ai.tools import ALL_TOOLS, create_tool_handler
from scd.models import DirMatch, DirMatchResult, RepoScanResult

logger = logging.getLogger(__name__)


async def match_directories(
    repo_a: RepoScanResult,
    repo_b: RepoScanResult,
    client: ClaudeClient,
) -> DirMatchResult:
    """Use Claude agent to explore both repos and match directories."""
    logger.info(
        "Matching directories — letting AI explore repos (%d dirs in A, %d dirs in B)",
        len(repo_a.dirs), len(repo_b.dirs),
    )

    tool_handler = create_tool_handler([repo_a.root_path, repo_b.root_path])

    user_msg = DIRECTORY_MATCH_USER.format(
        repo_a_path=repo_a.root_path,
        repo_b_path=repo_b.root_path,
    )

    data = await client.agent_loop_json(
        system=DIRECTORY_MATCH_SYSTEM,
        user=user_msg,
        tools=ALL_TOOLS,
        tool_handler=tool_handler,
    )

    result = DirMatchResult()

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

    matched_a = {m.dir_a for m in result.matched_dirs}
    matched_b = {m.dir_b for m in result.matched_dirs}

    result.orphan_dirs_a = [d for d in repo_a.dirs if d not in matched_a]
    result.orphan_dirs_b = [d for d in repo_b.dirs if d not in matched_b]

    logger.info(
        "Directory matching done: %d pairs, %d orphans in A, %d orphans in B",
        len(result.matched_dirs),
        len(result.orphan_dirs_a),
        len(result.orphan_dirs_b),
    )

    return result
