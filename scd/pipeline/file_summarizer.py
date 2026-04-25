"""Per-file summarizer (map step of the map-reduce summarizer).

For each source file in a repo we produce a small JSON summary describing
its purpose, exports, imports, frameworks, patterns, and a couple of
characteristic snippets. These per-file summaries are then consumed by the
directory-level reduce step (see ``scd.pipeline.dir_summarizer``).

Summaries are cached by `content_hash` in ``.scd_cache/file_summaries.jsonl``,
so identical contents (across files within one repo or across runs) only
ever get summarized once per (model, prompt) pair.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from scd.ai.client import LlmClient
from scd.ai.prompts import FILE_SUMMARY_SYSTEM, FILE_SUMMARY_USER
from scd.models import FileInfo, RepoScanResult
from scd.pipeline.dir_summarizer import (
    MAX_SINGLE_FILE_TOKENS,
    _count_tokens,
    _truncate_head_tail,
)

logger = logging.getLogger(__name__)

FILE_SUMMARY_CACHE_DIR_NAME = ".scd_cache"
FILE_SUMMARY_CACHE_FILE_NAME = "file_summaries.jsonl"
FILE_SUMMARY_CACHE_VERSION = 1
FILE_PROMPT_VERSION = 1

COMPACTION_RATIO = 4
COMPACTION_MIN_LINES = 50

# File summaries are deliberately compact (a few lines per field), so a small
# completion budget is plenty.
FILE_SUMMARY_OUTPUT_TOKENS = 1024

_PLACEHOLDER_FILE_SUMMARY = json.dumps(
    {
        "purpose": "unknown",
        "exports": [],
        "imports": [],
        "frameworks": [],
        "patterns": [],
        "key_snippets": [],
    },
    ensure_ascii=False,
)


def compute_file_hash(content: str, language: str) -> str:
    """Hash file content, salted with language and current cache/prompt versions.

    Same bytes interpreted as different languages would (in theory) get
    different summaries, so the language is part of the cache key.
    """
    h = hashlib.sha256()
    h.update(
        f"v{FILE_SUMMARY_CACHE_VERSION}|prompt={FILE_PROMPT_VERSION}".encode()
    )
    h.update(b"|lang=")
    h.update(language.encode())
    h.update(b"\x00")
    h.update(content.encode())
    return h.hexdigest()[:16]


class FileSummaryCache:
    """Append-only JSONL cache keyed by file content hash."""

    def __init__(self, repo_path: str, model: str) -> None:
        self._path = (
            Path(repo_path)
            / FILE_SUMMARY_CACHE_DIR_NAME
            / FILE_SUMMARY_CACHE_FILE_NAME
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._store: dict[str, dict] = {}
        self._raw_lines = 0
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> int:
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
                        "Skipping malformed file-summary cache line %d in %s: %s",
                        line_no, self._path, e,
                    )
                    continue
                if data.get("v") != FILE_SUMMARY_CACHE_VERSION:
                    mismatched += 1
                    continue
                if data.get("model") != self._model:
                    mismatched += 1
                    continue
                key = data.get("hash")
                if not key:
                    malformed += 1
                    continue
                self._store[key] = data
                self._raw_lines += 1
        if malformed:
            logger.warning(
                "%d malformed file-summary lines skipped in %s",
                malformed, self._path,
            )
        if mismatched:
            logger.info(
                "Ignored %d file-summary cache lines with stale version/model in %s",
                mismatched, self._path,
            )
        if (
            self._raw_lines > COMPACTION_RATIO * max(len(self._store), 1)
            and self._raw_lines > COMPACTION_MIN_LINES
        ):
            self._compact()
        return len(self._store)

    def get(self, content_hash: str) -> str | None:
        rec = self._store.get(content_hash)
        if rec is None:
            return None
        return rec.get("summary")

    async def put(
        self,
        content_hash: str,
        summary: str,
        *,
        language: str,
        path_hint: str,
    ) -> None:
        record = {
            "v": FILE_SUMMARY_CACHE_VERSION,
            "model": self._model,
            "hash": content_hash,
            "language": language,
            "summary": summary,
            "last_seen_path": path_hint,
        }
        async with self._lock:
            self._store[content_hash] = record
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            self._raw_lines += 1

    def _compact(self) -> None:
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


def _format_file_block(
    path: str, language: str, line_count: int, content: str,
) -> str:
    return (
        f"----- FILE: {path} ({language}, {line_count} lines) -----\n"
        f"{content}\n"
        f"----- END FILE -----"
    )


async def _summarize_file(
    file_info: FileInfo,
    content: str,
    client: LlmClient,
) -> tuple[str, dict[str, Any]]:
    """Run a single LLM call to summarize one file. Returns (json_str, stats)."""
    text, was_trunc = _truncate_head_tail(content or "", MAX_SINGLE_FILE_TOKENS)
    block = _format_file_block(
        file_info.path, file_info.language, file_info.line_count, text,
    )
    user_msg = FILE_SUMMARY_USER.format(
        file_path=file_info.path,
        language=file_info.language,
        line_count=file_info.line_count,
        file_block=block,
    )
    stats: dict[str, Any] = {
        "truncated": was_trunc,
        "tokens_input": _count_tokens(text),
    }
    try:
        result = await client.ask_json(
            system=FILE_SUMMARY_SYSTEM,
            user=user_msg,
            max_tokens=FILE_SUMMARY_OUTPUT_TOKENS,
        )
        return json.dumps(result, ensure_ascii=False), stats
    except Exception as e:
        logger.error("Failed to summarize file %s: %s", file_info.path, e)
        stats["error"] = f"{type(e).__name__}: {e}"
        return _PLACEHOLDER_FILE_SUMMARY, stats


async def summarize_files(
    repo: RepoScanResult,
    client: LlmClient,
    model: str,
) -> dict[str, str]:
    """Produce one JSON summary per source file in the repo.

    The returned dict maps each file path (as it appears in ``repo.dirs``) to
    its JSON-encoded summary string. Concurrency is bounded by the
    ``LlmClient``'s global in-flight semaphore.
    """
    cache = FileSummaryCache(repo.root_path, model)
    loaded = cache.load()
    if loaded:
        logger.info("Loaded %d cached file summaries from %s", loaded, cache.path)

    all_files: list[FileInfo] = []
    for dir_info in repo.dirs.values():
        all_files.extend(dir_info.files)

    summaries: dict[str, str] = {}
    cache_hits = 0
    by_hash: dict[str, list[tuple[FileInfo, str]]] = {}
    for f in all_files:
        content = repo.file_contents.get(f.path, "") or ""
        h = compute_file_hash(content, f.language)
        cached = cache.get(h)
        if cached is not None:
            summaries[f.path] = cached
            cache_hits += 1
            continue
        by_hash.setdefault(h, []).append((f, content))

    async def run_one(
        content_hash: str, fileset: list[tuple[FileInfo, str]],
    ) -> tuple[str, str, FileInfo, dict[str, Any]]:
        rep_file, content = fileset[0]
        summary, stats = await _summarize_file(rep_file, content, client)
        return content_hash, summary, rep_file, stats

    tasks = [
        asyncio.create_task(run_one(h, fileset))
        for h, fileset in by_hash.items()
    ]

    generated = 0
    errors = 0
    truncated: list[str] = []
    for task in asyncio.as_completed(tasks):
        content_hash, summary, rep_file, stats = await task
        if "error" in stats:
            errors += 1
            # Still propagate the placeholder to all paths sharing this hash,
            # but skip caching so the next run can retry.
            for f, _content in by_hash[content_hash]:
                summaries[f.path] = summary
            continue
        if stats.get("truncated"):
            truncated.append(rep_file.path)
        await cache.put(
            content_hash,
            summary,
            language=rep_file.language,
            path_hint=rep_file.path,
        )
        for f, _content in by_hash[content_hash]:
            summaries[f.path] = summary
        generated += 1

    logger.info(
        "File summaries for %s: %d files, %d generated, %d from cache, "
        "%d unique contents%s%s",
        repo.root_path,
        len(all_files),
        generated,
        cache_hits,
        len(by_hash),
        f", {errors} errors" if errors else "",
        f", truncated: {truncated}" if truncated else "",
    )
    return summaries
