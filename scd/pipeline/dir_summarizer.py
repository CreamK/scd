"""Hierarchical directory summarizer with caching.

Generates AI summaries bottom-up: leaf directories first, parent directories
after. For each directory, the LLM is given a ``read_file`` tool scoped to
the directory's **direct** files only, and the already-produced summaries
of its direct child directories are passed inline in the user prompt. The
LLM must read every direct file before its final JSON is accepted; child
directories are not inspected (their summaries are treated as authoritative).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from scd.ai.client import LlmClient
from scd.ai.prompts import DIR_SUMMARY_SYSTEM, DIR_SUMMARY_USER
from scd.ai.tools import DIR_SUMMARY_TOOLS, SubtreeFS
from scd.models import DirInfo, RepoScanResult

logger = logging.getLogger(__name__)

SUMMARY_CACHE_DIR_NAME = ".scd_cache"
SUMMARY_CACHE_FILE_NAME = "dir_summaries.jsonl"
SUMMARY_CACHE_VERSION = 4
PROMPT_VERSION = 1

COMPACTION_RATIO = 2
COMPACTION_MIN_LINES = 20

COVERAGE_THRESHOLD = 1.0
BASE_MAX_TOOL_TURNS = 100
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


class SummaryCache:
    """Append-only JSONL cache of directory summaries.

    One line per directory result. When the same dir appears on multiple lines,
    the last one wins. A cheap compaction is triggered at load time when the
    ratio of raw lines to unique dirs gets too high.
    """

    def __init__(self, repo_path: str, model: str) -> None:
        self._path = Path(repo_path) / SUMMARY_CACHE_DIR_NAME / SUMMARY_CACHE_FILE_NAME
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._store: dict[str, dict] = {}
        self._raw_lines = 0
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> int:
        """Populate in-memory store from disk. Returns unique dir count."""
        if not self._path.exists():
            return 0
        malformed = 0
        mismatched = 0
        with self._path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    malformed += 1
                    logger.warning(
                        "Skipping malformed cache line %d in %s: %s",
                        line_no, self._path, e,
                    )
                    continue
                if data.get("v") != SUMMARY_CACHE_VERSION:
                    mismatched += 1
                    continue
                if data.get("model") != self._model:
                    mismatched += 1
                    continue
                dir_path = data.get("dir")
                if dir_path is None:
                    malformed += 1
                    continue
                self._store[dir_path] = data
                self._raw_lines += 1
        if malformed:
            logger.warning("%d malformed lines skipped in %s", malformed, self._path)
        if mismatched:
            logger.info(
                "Ignored %d cache lines with stale version/model in %s",
                mismatched, self._path,
            )
        if (
            self._raw_lines > COMPACTION_RATIO * max(len(self._store), 1)
            and self._raw_lines > COMPACTION_MIN_LINES
        ):
            self._compact()
        return len(self._store)

    def get(self, dir_path: str) -> dict | None:
        return self._store.get(dir_path)

    async def put(self, dir_path: str, entry: dict) -> None:
        """Append one directory's result and update the in-memory store."""
        record = {
            "v": SUMMARY_CACHE_VERSION,
            "model": self._model,
            "dir": dir_path,
            **entry,
        }
        async with self._lock:
            self._store[dir_path] = record
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            self._raw_lines += 1

    def _compact(self) -> None:
        """Rewrite the file keeping only the latest entry per dir."""
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for record in self._store.values():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp, self._path)
        logger.info(
            "Compacted %s: %d raw lines -> %d unique entries",
            self._path, self._raw_lines, len(self._store),
        )
        self._raw_lines = len(self._store)


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
    """Target coverage for a directory.

    With the direct-files-only model, we always require the LLM to read
    every direct file. ``SubtreeFS.coverage`` short-circuits to 1.0 when
    ``total_files`` is 0, so empty directories pass without any read.
    """
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
        f"You have read {len(fs.read_paths)}/{fs.total_files} direct files "
        f"({fs.coverage:.0%}). Every direct file in this directory must be "
        f"read before the final JSON is accepted. Remaining unread direct "
        f"files (largest first): {unread_hint}. Call read_file on them and "
        f"then output ONLY the final JSON."
    )


async def _summarize_dir(
    dir_path: str,
    repo: RepoScanResult,
    child_summaries: dict[str, str],
    client: LlmClient,
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
    client: LlmClient,
    model: str,
) -> dict[str, str]:
    """Generate hierarchical summaries for all directories in a repo.

    Returns {dir_relative_path: summary_json_string}.
    Uses cache to skip directories whose subtree hasn't changed.
    """
    cache = SummaryCache(repo.root_path, model)
    loaded = cache.load()
    if loaded:
        logger.info("Loaded %d cached dir summaries from %s", loaded, cache.path)
    summaries: dict[str, str] = {}
    dir_stats: dict[str, dict[str, Any]] = {}
    levels = _build_tree_levels(repo)

    cache_hits = 0
    generated = 0

    for level_dirs in levels:
        tasks: list[tuple[str, asyncio.Task | None]] = []

        for dir_path in level_dirs:
            content_hash = compute_subtree_hash(dir_path, repo)
            cached_entry = cache.get(dir_path)
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

            if "error" in stats:
                logger.warning(
                    "Skip cache write for %s due to summarize error; "
                    "keep previous entry if any",
                    dir_path or "(root)",
                )
                continue

            content_hash = compute_subtree_hash(dir_path, repo)
            await cache.put(dir_path, {
                "content_hash": content_hash,
                "summary": summary_json,
                "coverage": stats.get("coverage"),
                "files_read": stats.get("files_read", 0),
                "total_files": stats.get("total_files", 0),
            })

            coverage = stats.get("coverage")
            coverage_txt = f"{coverage:.0%}" if isinstance(coverage, (int, float)) else "n/a"
            logger.info(
                "summarized %s: read %d/%d files (%s)",
                dir_path or "(root)",
                stats.get("files_read", 0),
                stats.get("total_files", 0),
                coverage_txt,
            )

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
