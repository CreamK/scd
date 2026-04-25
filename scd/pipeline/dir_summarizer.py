"""Hierarchical directory summarizer with caching.

Generates AI summaries bottom-up: leaf directories first, parent directories
after. For each directory we feed the LLM the **full source** of every direct
file together with the already-produced summaries of direct child directories
in a single prompt - no tool-use loop. When the combined size exceeds the
context budget we bin-pack files into chunks, produce one partial summary per
chunk, then run a final merge call that combines partials with child
summaries into the final JSON.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import tiktoken

from scd.ai.client import LlmClient
from scd.ai.prompts import (
    DIR_SUMMARY_MERGE_SYSTEM,
    DIR_SUMMARY_MERGE_USER,
    DIR_SUMMARY_PARTIAL_SYSTEM,
    DIR_SUMMARY_PARTIAL_USER,
    DIR_SUMMARY_SYSTEM,
    DIR_SUMMARY_USER,
)
from scd.models import DirInfo, FileInfo, RepoScanResult

logger = logging.getLogger(__name__)

SUMMARY_CACHE_DIR_NAME = ".scd_cache"
SUMMARY_CACHE_FILE_NAME = "dir_summaries.jsonl"
SUMMARY_CACHE_VERSION = 4
PROMPT_VERSION = 1

COMPACTION_RATIO = 2
COMPACTION_MIN_LINES = 20

# Token budget (assumes a 128k-context model). We deliberately cap usable
# input at 96k (75% of the physical window) to stay safe against:
# - cross-tokenizer skew (cl100k_base vs whatever the deployed model uses,
#   typically +5-10% on code/CJK content),
# - implicit chat-template / schema overhead on the server side,
# - quality degradation ("lost in the middle") on long-context models.
MAX_CONTEXT_TOKENS = 128_000
OUTPUT_RESERVE_TOKENS = 8_192        # matches ask_json(max_tokens=8192)
SAFETY_MARGIN_TOKENS = 16_808        # ~14% of the context; covers tokenizer skew + gateway slack
PROMPT_OVERHEAD_TOKENS = 7_000       # system prompt + JSON schema + file markers + child summaries
INPUT_BUDGET_TOKENS = (
    MAX_CONTEXT_TOKENS
    - OUTPUT_RESERVE_TOKENS
    - SAFETY_MARGIN_TOKENS
    - PROMPT_OVERHEAD_TOKENS
)  # = 96_000
MAX_SINGLE_FILE_TOKENS = 32_000      # head + tail each ~16k; ~1/3 of the input budget
TIKTOKEN_ENCODING = "cl100k_base"

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


# --------------------------------------------------------------------------
# Token utilities
# --------------------------------------------------------------------------

_ENCODER: tiktoken.Encoding | None = None

# Vendored tiktoken BPE cache so we don't have to hit
# openaipublic.blob.core.windows.net at runtime. The filename is the SHA1 of
# the download URL tiktoken would otherwise use.
_VENDORED_TIKTOKEN_DIR = Path(__file__).resolve().parents[1] / "vendor" / "tiktoken"


def _get_encoder() -> tiktoken.Encoding:
    """Lazy module-level encoder singleton.

    Points ``TIKTOKEN_CACHE_DIR`` at our vendored cache before loading so the
    encoder is initialized fully offline. An explicit user-set
    ``TIKTOKEN_CACHE_DIR`` wins.
    """
    global _ENCODER
    if _ENCODER is None:
        if (
            "TIKTOKEN_CACHE_DIR" not in os.environ
            and _VENDORED_TIKTOKEN_DIR.is_dir()
        ):
            os.environ["TIKTOKEN_CACHE_DIR"] = str(_VENDORED_TIKTOKEN_DIR)
        _ENCODER = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return _ENCODER


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_encoder().encode(text, disallowed_special=()))


def _truncate_head_tail(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate ``text`` to roughly ``max_tokens`` tokens, keeping head + tail.

    Returns ``(new_text, was_truncated)``. If the input already fits, the
    original string is returned unchanged.
    """
    if max_tokens <= 0 or not text:
        return text, False
    enc = _get_encoder()
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text, False
    half = max(1, max_tokens // 2)
    head = enc.decode(tokens[:half])
    tail = enc.decode(tokens[-half:])
    dropped = len(tokens) - 2 * half
    marker = f"\n\n... [truncated {dropped} tokens] ...\n\n"
    return head + marker + tail, True


# --------------------------------------------------------------------------
# Cache + tree helpers (unchanged behaviour)
# --------------------------------------------------------------------------


def compute_subtree_hash(dir_path: str, repo: RepoScanResult) -> str:
    """Hash every file (path + content) inside the subtree rooted at dir_path.

    Any change to any descendant file invalidates the cached summary. The
    parent directory's summary depends transitively on its child summaries,
    which in turn depend on their own subtrees, so a subtree hash is the
    correct invalidation key for the parent as well.
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


# --------------------------------------------------------------------------
# Prompt formatting helpers
# --------------------------------------------------------------------------


def _format_direct_dirs(dirs: list[str]) -> str:
    if not dirs:
        return "(none)"
    return ", ".join(dirs)


def _format_child_summaries(child_summaries: dict[str, str]) -> str:
    if not child_summaries:
        return "(no child directories)"
    lines: list[str] = []
    for child_dir in sorted(child_summaries.keys()):
        lines.append(f"  [{child_dir}]: {child_summaries[child_dir]}")
    return "\n".join(lines)


def _format_file_block(path: str, language: str, line_count: int, content: str) -> str:
    """Render one file as a fenced block for inclusion in user prompts."""
    return (
        f"----- FILE: {path} ({language}, {line_count} lines) -----\n"
        f"{content}\n"
        f"----- END FILE -----"
    )


def _format_files_block(prepared: list["_PreparedFile"]) -> str:
    if not prepared:
        return "(no direct files in this directory)"
    return "\n\n".join(
        _format_file_block(p.display_path, p.language, p.line_count, p.text)
        for p in prepared
    )


def _format_partial_summaries(partials: list[str]) -> str:
    if not partials:
        return "(none)"
    blocks: list[str] = []
    for i, summary in enumerate(partials, 1):
        blocks.append(f"[Chunk {i}]\n{summary}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Direct file preparation + chunking
# --------------------------------------------------------------------------


class _PreparedFile:
    """A direct file ready to be embedded into a prompt."""

    __slots__ = (
        "path",
        "display_path",
        "language",
        "line_count",
        "text",
        "tokens",
        "truncated",
    )

    def __init__(
        self,
        *,
        path: str,
        display_path: str,
        language: str,
        line_count: int,
        text: str,
        tokens: int,
        truncated: bool,
    ) -> None:
        self.path = path
        self.display_path = display_path
        self.language = language
        self.line_count = line_count
        self.text = text
        self.tokens = tokens
        self.truncated = truncated


def _prepare_direct_files(
    dir_path: str, repo: RepoScanResult,
) -> tuple[list[_PreparedFile], list[str]]:
    """Read content for each direct file of ``dir_path`` and pre-tokenize.

    Files larger than ``MAX_SINGLE_FILE_TOKENS`` are head + tail truncated.
    Returns ``(prepared_files, truncated_paths)``.
    """
    dir_info = repo.dirs.get(dir_path)
    files: list[FileInfo] = list(dir_info.files) if dir_info is not None else []
    files.sort(key=lambda f: f.path)

    prefix = f"{dir_path}/" if dir_path else ""
    prepared: list[_PreparedFile] = []
    truncated_paths: list[str] = []

    for f in files:
        raw = repo.file_contents.get(f.path, "") or ""
        text, was_trunc = _truncate_head_tail(raw, MAX_SINGLE_FILE_TOKENS)
        tokens = _count_tokens(text)
        display_path = f.path[len(prefix):] if (prefix and f.path.startswith(prefix)) else f.path
        prepared.append(_PreparedFile(
            path=f.path,
            display_path=display_path,
            language=f.language,
            line_count=f.line_count,
            text=text,
            tokens=tokens,
            truncated=was_trunc,
        ))
        if was_trunc:
            truncated_paths.append(f.path)

    return prepared, truncated_paths


def _bin_pack_by_tokens(
    items: list[_PreparedFile], budget: int,
) -> list[list[_PreparedFile]]:
    """First-fit decreasing bin-packing by ``tokens``.

    Items larger than ``budget`` get their own bin (truncation should have
    prevented this, but we don't drop content silently).
    """
    sorted_items = sorted(items, key=lambda x: x.tokens, reverse=True)
    bins: list[list[_PreparedFile]] = []
    sizes: list[int] = []
    for item in sorted_items:
        placed = False
        for i, size in enumerate(sizes):
            if size + item.tokens <= budget:
                bins[i].append(item)
                sizes[i] += item.tokens
                placed = True
                break
        if not placed:
            bins.append([item])
            sizes.append(item.tokens)
    # Stabilize order inside each bin by path for deterministic prompts.
    for b in bins:
        b.sort(key=lambda x: x.display_path)
    return bins


# --------------------------------------------------------------------------
# Single-directory summarization
# --------------------------------------------------------------------------


async def _summarize_dir(
    dir_path: str,
    repo: RepoScanResult,
    child_summaries: dict[str, str],
    client: LlmClient,
) -> tuple[str, dict[str, Any]]:
    """Summarize a single directory by feeding direct file contents to the LLM.

    Returns ``(summary_json_str, stats_dict)``. Picks single-shot when
    everything fits into the input budget, otherwise bin-packs into chunks
    and runs a partial-then-merge flow.
    """
    prepared, truncated_paths = _prepare_direct_files(dir_path, repo)
    total_files = len(prepared)
    total_lines = sum(p.line_count for p in prepared)
    file_tokens = sum(p.tokens for p in prepared)

    direct_children = _get_direct_children(dir_path, repo.dirs)
    direct_dirs_txt = _format_direct_dirs(direct_children)
    child_summaries_txt = _format_child_summaries(child_summaries)
    child_summaries_tokens = _count_tokens(child_summaries_txt)

    if child_summaries_tokens > INPUT_BUDGET_TOKENS // 2:
        logger.warning(
            "Child summaries for %s are unusually large (%d tokens); "
            "merge prompt may be tight against the context window",
            dir_path or "(root)", child_summaries_tokens,
        )

    stats: dict[str, Any] = {
        "coverage": 1.0,
        "files_read": total_files,
        "total_files": total_files,
        "chunks": 1,
        "truncated_files": truncated_paths,
        "tokens_input": file_tokens + child_summaries_tokens,
    }

    single_shot_total = file_tokens + child_summaries_tokens
    fits_single_shot = single_shot_total <= INPUT_BUDGET_TOKENS

    try:
        if fits_single_shot:
            result = await _single_shot(
                dir_path=dir_path,
                prepared=prepared,
                total_files=total_files,
                total_lines=total_lines,
                direct_dirs_txt=direct_dirs_txt,
                child_summaries_txt=child_summaries_txt,
                client=client,
            )
        else:
            chunks = _bin_pack_by_tokens(prepared, INPUT_BUDGET_TOKENS)
            stats["chunks"] = len(chunks)
            result = await _chunked_merge(
                dir_path=dir_path,
                chunks=chunks,
                direct_dirs_txt=direct_dirs_txt,
                child_summaries_txt=child_summaries_txt,
                client=client,
            )
        return json.dumps(result, ensure_ascii=False), stats
    except Exception as e:
        logger.error(
            "Failed to summarize dir %s (%d direct files, %d tokens): %s",
            dir_path or "(root)", total_files, single_shot_total, e,
        )
        stats["error"] = f"{type(e).__name__}: {e}"
        return _PLACEHOLDER_SUMMARY, stats


async def _single_shot(
    *,
    dir_path: str,
    prepared: list[_PreparedFile],
    total_files: int,
    total_lines: int,
    direct_dirs_txt: str,
    child_summaries_txt: str,
    client: LlmClient,
) -> dict:
    user_msg = DIR_SUMMARY_USER.format(
        dir_path=dir_path or "(root)",
        direct_dirs=direct_dirs_txt,
        child_summaries=child_summaries_txt,
        total_files=total_files,
        total_lines=total_lines,
        files_block=_format_files_block(prepared),
    )
    return await client.ask_json(system=DIR_SUMMARY_SYSTEM, user=user_msg)


async def _chunked_merge(
    *,
    dir_path: str,
    chunks: list[list[_PreparedFile]],
    direct_dirs_txt: str,
    child_summaries_txt: str,
    client: LlmClient,
) -> dict:
    chunk_total = len(chunks)
    partial_coros = [
        _partial_for_chunk(
            dir_path=dir_path,
            chunk=chunk,
            chunk_index=i + 1,
            chunk_total=chunk_total,
            client=client,
        )
        for i, chunk in enumerate(chunks)
    ]
    partial_results = await asyncio.gather(*partial_coros)

    merge_user = DIR_SUMMARY_MERGE_USER.format(
        dir_path=dir_path or "(root)",
        direct_dirs=direct_dirs_txt,
        child_summaries=child_summaries_txt,
        chunk_total=chunk_total,
        partial_summaries=_format_partial_summaries(partial_results),
    )
    return await client.ask_json(system=DIR_SUMMARY_MERGE_SYSTEM, user=merge_user)


async def _partial_for_chunk(
    *,
    dir_path: str,
    chunk: list[_PreparedFile],
    chunk_index: int,
    chunk_total: int,
    client: LlmClient,
) -> str:
    chunk_files = len(chunk)
    chunk_lines = sum(p.line_count for p in chunk)
    user_msg = DIR_SUMMARY_PARTIAL_USER.format(
        dir_path=dir_path or "(root)",
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        chunk_files=chunk_files,
        chunk_lines=chunk_lines,
        files_block=_format_files_block(chunk),
    )
    result = await client.ask_json(system=DIR_SUMMARY_PARTIAL_SYSTEM, user=user_msg)
    return json.dumps(result, ensure_ascii=False)


# --------------------------------------------------------------------------
# Repo-wide orchestration
# --------------------------------------------------------------------------


async def summarize_repo(
    repo: RepoScanResult,
    client: LlmClient,
    model: str,
) -> dict[str, str]:
    """Generate hierarchical summaries for all directories in a repo.

    Returns ``{dir_relative_path: summary_json_string}``. Uses a per-subtree
    content hash to skip directories whose subtree hasn't changed.
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
                    "chunks": cached_entry.get("chunks", 1),
                    "truncated_files": cached_entry.get("truncated_files", []),
                    "tokens_input": cached_entry.get("tokens_input"),
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
                "chunks": stats.get("chunks", 1),
                "truncated_files": stats.get("truncated_files", []),
                "tokens_input": stats.get("tokens_input"),
            })

            chunks = stats.get("chunks", 1)
            chunk_suffix = "" if chunks == 1 else f", chunked: {chunks}+1 calls"
            trunc = stats.get("truncated_files") or []
            trunc_suffix = "" if not trunc else f", truncated: {trunc}"
            logger.info(
                "summarized %s: %d direct files, %s tokens%s%s",
                dir_path or "(root)",
                stats.get("total_files", 0),
                stats.get("tokens_input", "?"),
                chunk_suffix,
                trunc_suffix,
            )

    logger.info(
        "Summarized %d directories: %d generated, %d from cache",
        len(summaries), generated, cache_hits,
    )
    return summaries
