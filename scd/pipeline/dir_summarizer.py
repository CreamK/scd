"""Hierarchical directory summarizer with caching.

Generates AI summaries bottom-up: leaf directories first (from code),
then parent directories (from child summaries + own files).
"""

from __future__ import annotations

import asyncio
import json
import logging

from scd.ai.client import ClaudeClient
from scd.ai.prompts import (
    DIR_SUMMARY_LEAF_SYSTEM,
    DIR_SUMMARY_LEAF_USER,
    DIR_SUMMARY_PARENT_SYSTEM,
    DIR_SUMMARY_PARENT_USER,
)
from scd.cache import (
    compute_dir_hash,
    compute_parent_hash,
    load_cache,
    save_cache,
)
from scd.models import DirInfo, RepoScanResult

logger = logging.getLogger(__name__)

MAX_LINES_PER_FILE = 200


def _build_tree_levels(repo: RepoScanResult) -> list[list[str]]:
    """Build directory tree levels for bottom-up traversal.

    Returns a list of levels, where level 0 contains leaf directories
    and the last level contains root-level directories.
    """
    dirs = repo.dirs
    children: dict[str, list[str]] = {d: [] for d in dirs}

    for dir_path in dirs:
        if not dir_path:
            continue
        parent = dir_path.rsplit("/", 1)[0] if "/" in dir_path else ""
        if parent in children:
            children[parent].append(dir_path)

    depth: dict[str, int] = {}

    def _get_depth(d: str) -> int:
        if d in depth:
            return depth[d]
        if not children[d]:
            depth[d] = 0
            return 0
        depth[d] = 1 + max(_get_depth(c) for c in children[d])
        return depth[d]

    for d in dirs:
        _get_depth(d)

    max_depth = max(depth.values()) if depth else 0
    levels: list[list[str]] = [[] for _ in range(max_depth + 1)]
    for d, dep in depth.items():
        levels[dep].append(d)

    return levels


def _get_direct_children(dir_path: str, all_dirs: dict[str, DirInfo]) -> list[str]:
    """Get immediate child directories of a directory."""
    result = []
    prefix = f"{dir_path}/" if dir_path else ""
    for d in all_dirs:
        if not d:
            continue
        if prefix and not d.startswith(prefix):
            continue
        if not prefix and "/" in d:
            continue
        if prefix:
            remainder = d[len(prefix):]
        else:
            remainder = d
        if "/" not in remainder:
            result.append(d)
    return sorted(result)


def _format_file_contents(dir_info: DirInfo, file_contents: dict[str, str]) -> str:
    """Format all files in a directory for the prompt."""
    parts: list[str] = []
    for f in sorted(dir_info.files, key=lambda x: x.path):
        content = file_contents.get(f.path, "")
        lines = content.splitlines()
        if len(lines) > MAX_LINES_PER_FILE:
            truncated = "\n".join(lines[:MAX_LINES_PER_FILE])
            truncated += f"\n\n... ({len(lines) - MAX_LINES_PER_FILE} more lines, {len(lines)} total)"
        else:
            truncated = content
        parts.append(f"--- {f.path} ({f.language}, {f.line_count} lines) ---\n{truncated}")
    return "\n\n".join(parts) if parts else "(no direct source files)"


async def _summarize_leaf(
    dir_path: str,
    dir_info: DirInfo,
    file_contents: dict[str, str],
    client: ClaudeClient,
) -> str:
    """Generate summary for a leaf directory (no child directories with source files)."""
    user_msg = DIR_SUMMARY_LEAF_USER.format(
        dir_path=dir_path or "(root)",
        file_count=len(dir_info.files),
        file_contents=_format_file_contents(dir_info, file_contents),
    )
    try:
        data = await client.ask_json(system=DIR_SUMMARY_LEAF_SYSTEM, user=user_msg)
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to summarize leaf dir %s: %s", dir_path, e)
        return json.dumps({"purpose": "unknown", "key_exports": [], "frameworks": [], "patterns": []})


async def _summarize_parent(
    dir_path: str,
    dir_info: DirInfo,
    file_contents: dict[str, str],
    child_summaries: dict[str, str],
    client: ClaudeClient,
) -> str:
    """Generate summary for a parent directory using child summaries + own files."""
    child_parts = []
    for child_dir in sorted(child_summaries.keys()):
        child_parts.append(f"  [{child_dir}]: {child_summaries[child_dir]}")
    child_text = "\n".join(child_parts) if child_parts else "(no child directories)"

    user_msg = DIR_SUMMARY_PARENT_USER.format(
        dir_path=dir_path or "(root)",
        child_summaries=child_text,
        file_count=len(dir_info.files),
        file_contents=_format_file_contents(dir_info, file_contents),
    )
    try:
        data = await client.ask_json(system=DIR_SUMMARY_PARENT_SYSTEM, user=user_msg)
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to summarize parent dir %s: %s", dir_path, e)
        return json.dumps({"purpose": "unknown", "key_exports": [], "frameworks": [], "patterns": [], "children_overview": ""})


async def summarize_repo(
    repo: RepoScanResult,
    client: ClaudeClient,
    model: str,
) -> dict[str, str]:
    """Generate hierarchical summaries for all directories in a repo.

    Returns {dir_relative_path: summary_json_string}.
    Uses cache to skip directories that haven't changed.
    """
    cached = load_cache(repo.root_path, model)
    summaries: dict[str, str] = {}
    levels = _build_tree_levels(repo)

    cache_hits = 0
    generated = 0

    for level_idx, level_dirs in enumerate(levels):
        tasks: list[tuple[str, asyncio.Task | None]] = []

        for dir_path in level_dirs:
            dir_info = repo.dirs[dir_path]
            children = _get_direct_children(dir_path, repo.dirs)

            if children:
                child_sums = {c: summaries[c] for c in children if c in summaries}
                content_hash = compute_parent_hash(dir_info, repo.file_contents, child_sums)
            else:
                content_hash = compute_dir_hash(dir_info, repo.file_contents)

            cached_entry = cached.get(dir_path)
            if cached_entry and cached_entry.get("content_hash") == content_hash:
                summaries[dir_path] = cached_entry["summary"]
                cache_hits += 1
                tasks.append((dir_path, None))
                continue

            if children:
                child_sums = {c: summaries[c] for c in children if c in summaries}
                coro = _summarize_parent(dir_path, dir_info, repo.file_contents, child_sums, client)
            else:
                coro = _summarize_leaf(dir_path, dir_info, repo.file_contents, client)

            task = asyncio.create_task(coro)
            tasks.append((dir_path, task))

        for dir_path, task in tasks:
            if task is not None:
                summaries[dir_path] = await task
                generated += 1

                dir_info = repo.dirs[dir_path]
                children = _get_direct_children(dir_path, repo.dirs)
                if children:
                    child_sums = {c: summaries[c] for c in children if c in summaries}
                    content_hash = compute_parent_hash(dir_info, repo.file_contents, child_sums)
                else:
                    content_hash = compute_dir_hash(dir_info, repo.file_contents)

                cached[dir_path] = {"content_hash": content_hash, "summary": summaries[dir_path]}

    save_cache(repo.root_path, model, cached)

    logger.info(
        "Summarized %d directories: %d generated, %d from cache",
        len(summaries), generated, cache_hits,
    )
    return summaries
