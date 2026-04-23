"""Hierarchical directory summarizer with caching.

Generates AI summaries bottom-up: leaf directories first, parent directories
after. The LLM is given `list_dir` / `read_file` tools scoped to each
directory's subtree and must reach a minimum file-read coverage before its
final JSON is accepted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

from scd.ai.client import ClaudeClient
from scd.ai.prompts import DIR_SUMMARY_SYSTEM, DIR_SUMMARY_USER
from scd.ai.tools import DIR_SUMMARY_TOOLS, SubtreeFS
from scd.models import DirInfo, RepoScanResult

logger = logging.getLogger(__name__)

SUMMARY_CACHE_DIR_NAME = ".scd_cache"
SUMMARY_CACHE_FILE_NAME = "dir_summaries.json"
SUMMARY_CACHE_VERSION = 2
PROMPT_VERSION = 1

COVERAGE_THRESHOLD = 0.7
SMALL_DIR_FORCE_FULL = 3
BASE_MAX_TOOL_TURNS = 20
FOLLOW_UP_TOP_K_BUFFER = 2
UNREAD_HINT_MAX = 15

_PLACEHOLDER_SUMMARY = json.dumps(
    {
        "purpose": "unknown",
        "key_exports": [],
        "frameworks": [],
        "patterns": [],
        "children_overview": "",
    },
    ensure_ascii=False,
)


def compute_subtree_hash(dir_path: str, repo: RepoScanResult) -> str:
    """Hash every file (path + content) inside the subtree rooted at dir_path.

    Any change to any descendant file invalidates the cached summary.
    """
    h = hashlib.sha256()
    h.update(f"v{SUMMARY_CACHE_VERSION}|prompt={PROMPT_VERSION}".encode())
    h.update(b"|root=")
    h.update(dir_path.encode())

    prefix = f"{dir_path}/" if dir_path else ""
    subtree_files: list[tuple[str, str]] = []
    for d, dir_info in repo.dirs.items():
        if dir_path:
            if d != dir_path and not d.startswith(prefix):
                continue
        for f in dir_info.files:
            subtree_files.append((f.path, repo.file_contents.get(f.path, "")))
    for path, content in sorted(subtree_files, key=lambda x: x[0]):
        h.update(b"\x00")
        h.update(path.encode())
        h.update(b"\x00")
        h.update(content.encode())
    return h.hexdigest()[:16]


def load_cache(repo_path: str, model: str) -> dict[str, dict]:
    """Load cached summaries from repo's .scd_cache directory.

    Returns dict: {dir_path: {"content_hash": ..., "summary": ..., ...}}
    """
    cache_path = Path(repo_path) / SUMMARY_CACHE_DIR_NAME / SUMMARY_CACHE_FILE_NAME
    if not cache_path.exists():
        return {}

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read cache at %s: %s", cache_path, e)
        return {}

    if data.get("version") != SUMMARY_CACHE_VERSION or data.get("model") != model:
        logger.info("Cache invalidated (version/model mismatch)")
        return {}

    return data.get("directories", {})


def save_cache(repo_path: str, model: str, directories: dict[str, dict]) -> str:
    """Save directory summaries to repo's .scd_cache directory. Returns cache file path."""
    cache_dir = Path(repo_path) / SUMMARY_CACHE_DIR_NAME
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / SUMMARY_CACHE_FILE_NAME

    data = {
        "version": SUMMARY_CACHE_VERSION,
        "model": model,
        "directories": directories,
    }
    cache_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(cache_path)


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


def _target_coverage_for(fs: SubtreeFS) -> float:
    if fs.total_files == 0:
        return 1.0
    if fs.total_files <= SMALL_DIR_FORCE_FULL:
        return 1.0
    return COVERAGE_THRESHOLD


def _target_files_for(fs: SubtreeFS, target_coverage: float) -> int:
    if fs.total_files == 0:
        return 0
    return max(1, math.ceil(fs.total_files * target_coverage))


def _format_inventory_dirs(dirs: list[str]) -> str:
    if not dirs:
        return "(none)"
    return ", ".join(dirs)


def _format_inventory_files(files: list[dict[str, Any]]) -> str:
    if not files:
        return "(none)"
    return ", ".join(f"{f['path']} ({f['language']}, {f['line_count']} lines)" for f in files)


def _format_child_summaries(child_summaries: dict[str, str]) -> str:
    if not child_summaries:
        return "(no child directories)"
    lines: list[str] = []
    for child_dir in sorted(child_summaries.keys()):
        lines.append(f"  [{child_dir}]: {child_summaries[child_dir]}")
    return "\n".join(lines)


def _build_follow_up(fs: SubtreeFS, target_coverage: float) -> str:
    target_files = _target_files_for(fs, target_coverage)
    still_needed = max(0, target_files - len(fs.read_paths))
    top_k = still_needed + FOLLOW_UP_TOP_K_BUFFER
    unread = fs.unread_files_ranked()[:top_k]
    unread_hint = (
        ", ".join(f"{f.path} ({f.line_count} lines)" for f in unread)
        if unread
        else "(none)"
    )
    return (
        f"Your file coverage is {fs.coverage:.0%} "
        f"({len(fs.read_paths)}/{fs.total_files}). The required minimum is "
        f"{target_coverage:.0%}. You MUST call read_file on additional files "
        f"before giving the final JSON. Suggested unread files (largest first): "
        f"{unread_hint}. After reading them, output ONLY the final JSON."
    )


async def _summarize_dir(
    dir_path: str,
    repo: RepoScanResult,
    child_summaries: dict[str, str],
    client: ClaudeClient,
) -> tuple[str, dict[str, Any]]:
    """Summarize a single directory using tool-use with a coverage validator.

    Returns (summary_json_str, stats_dict).
    """
    fs = SubtreeFS(root_rel=dir_path, repo=repo)
    target_coverage = _target_coverage_for(fs)
    target_files = _target_files_for(fs, target_coverage)

    max_tool_turns = max(BASE_MAX_TOOL_TURNS, target_files + 5)

    inventory = fs.list_dir("")
    direct_dirs_txt = _format_inventory_dirs(inventory.get("dirs", []) or [])
    direct_files_txt = _format_inventory_files(inventory.get("files", []) or [])

    user_msg = DIR_SUMMARY_USER.format(
        dir_path=dir_path or "(root)",
        direct_dirs=direct_dirs_txt,
        direct_files=direct_files_txt,
        total_files=fs.total_files,
        total_lines=fs.total_lines,
        child_summaries=_format_child_summaries(child_summaries),
    )

    async def tool_handler(name: str, tool_input: dict[str, Any]) -> Any:
        if name == "list_dir":
            return fs.list_dir(tool_input.get("path", "") or "")
        if name == "read_file":
            return fs.read_file(
                tool_input.get("path", "") or "",
                offset=tool_input.get("offset", 0) or 0,
                limit=tool_input.get("limit", None),
            )
        return {"error": f"unknown tool: {name}"}

    def validator(_result: dict) -> tuple[bool, str | None]:
        if fs.coverage >= target_coverage - 1e-9:
            return True, None
        return False, _build_follow_up(fs, target_coverage)

    stats: dict[str, Any] = {
        "coverage": None,
        "files_read": 0,
        "total_files": fs.total_files,
        "target_coverage": target_coverage,
    }

    try:
        result = await client.ask_json_with_tools(
            system=DIR_SUMMARY_SYSTEM,
            user=user_msg,
            tools=DIR_SUMMARY_TOOLS,
            tool_handler=tool_handler,
            max_tool_turns=max_tool_turns,
            validator=validator,
        )
        stats["coverage"] = fs.coverage
        stats["files_read"] = len(fs.read_paths)
        return json.dumps(result, ensure_ascii=False), stats
    except Exception as e:
        logger.error(
            "Failed to summarize dir %s (coverage=%.0f%%, %d/%d files read): %s",
            dir_path or "(root)", fs.coverage * 100,
            len(fs.read_paths), fs.total_files, e,
        )
        stats["coverage"] = fs.coverage if fs.total_files else None
        stats["files_read"] = len(fs.read_paths)
        stats["error"] = f"{type(e).__name__}: {e}"
        return _PLACEHOLDER_SUMMARY, stats


async def summarize_repo(
    repo: RepoScanResult,
    client: ClaudeClient,
    model: str,
) -> dict[str, str]:
    """Generate hierarchical summaries for all directories in a repo.

    Returns {dir_relative_path: summary_json_string}.
    Uses cache to skip directories whose subtree hasn't changed.
    """
    cached = load_cache(repo.root_path, model)
    summaries: dict[str, str] = {}
    dir_stats: dict[str, dict[str, Any]] = {}
    levels = _build_tree_levels(repo)

    cache_hits = 0
    generated = 0

    for level_dirs in levels:
        tasks: list[tuple[str, asyncio.Task | None]] = []

        for dir_path in level_dirs:
            content_hash = compute_subtree_hash(dir_path, repo)
            cached_entry = cached.get(dir_path)
            if cached_entry and cached_entry.get("content_hash") == content_hash:
                summaries[dir_path] = cached_entry["summary"]
                dir_stats[dir_path] = {
                    "coverage": cached_entry.get("coverage"),
                    "files_read": cached_entry.get("files_read", 0),
                    "total_files": cached_entry.get("total_files", 0),
                    "cached": True,
                }
                cache_hits += 1
                tasks.append((dir_path, None))
                continue

            children = _get_direct_children(dir_path, repo.dirs)
            child_sums = {c: summaries[c] for c in children if c in summaries}
            coro = _summarize_dir(dir_path, repo, child_sums, client)
            task = asyncio.create_task(coro)
            tasks.append((dir_path, task))

        for dir_path, task in tasks:
            if task is None:
                continue
            summary_json, stats = await task
            summaries[dir_path] = summary_json
            dir_stats[dir_path] = stats
            generated += 1

            content_hash = compute_subtree_hash(dir_path, repo)
            cached[dir_path] = {
                "content_hash": content_hash,
                "summary": summary_json,
                "coverage": stats.get("coverage"),
                "files_read": stats.get("files_read", 0),
                "total_files": stats.get("total_files", 0),
            }

            coverage = stats.get("coverage")
            coverage_txt = f"{coverage:.0%}" if isinstance(coverage, (int, float)) else "n/a"
            logger.info(
                "summarized %s: read %d/%d files (%s)",
                dir_path or "(root)",
                stats.get("files_read", 0),
                stats.get("total_files", 0),
                coverage_txt,
            )

    save_cache(repo.root_path, model, cached)

    coverages = [
        s["coverage"] for s in dir_stats.values()
        if isinstance(s.get("coverage"), (int, float))
    ]
    below = [
        d for d, s in dir_stats.items()
        if isinstance(s.get("coverage"), (int, float))
        and s["coverage"] < COVERAGE_THRESHOLD
        and s.get("total_files", 0) > 0
    ]
    if coverages:
        logger.info(
            "coverage stats: min=%.0f%%, avg=%.0f%%, below-threshold dirs=%s",
            min(coverages) * 100,
            (sum(coverages) / len(coverages)) * 100,
            [d or "(root)" for d in below],
        )

    logger.info(
        "Summarized %d directories: %d generated, %d from cache",
        len(summaries), generated, cache_hits,
    )
    return summaries
