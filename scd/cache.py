"""Directory summary cache — stores per-directory AI summaries keyed by content hash."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from scd.models import DirInfo, RepoScanResult

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".scd_cache"
CACHE_FILE_NAME = "dir_summaries.json"
CACHE_VERSION = 1


def compute_dir_hash(dir_info: DirInfo, file_contents: dict[str, str]) -> str:
    """Compute a stable hash for a directory based on its file paths and contents."""
    h = hashlib.sha256()
    for f in sorted(dir_info.files, key=lambda x: x.path):
        h.update(f.path.encode())
        content = file_contents.get(f.path, "")
        h.update(content.encode())
    return h.hexdigest()[:16]


def compute_parent_hash(
    dir_info: DirInfo,
    file_contents: dict[str, str],
    child_summaries: dict[str, str],
) -> str:
    """Compute hash for a parent directory: own files + child summary content."""
    h = hashlib.sha256()
    for f in sorted(dir_info.files, key=lambda x: x.path):
        h.update(f.path.encode())
        content = file_contents.get(f.path, "")
        h.update(content.encode())
    for child_dir in sorted(child_summaries.keys()):
        h.update(child_dir.encode())
        h.update(child_summaries[child_dir].encode())
    return h.hexdigest()[:16]


def load_cache(repo_path: str, model: str) -> dict[str, dict]:
    """Load cached summaries from repo's .scd_cache directory.

    Returns dict: {dir_path: {"content_hash": ..., "summary": ...}}
    """
    cache_path = Path(repo_path) / CACHE_DIR_NAME / CACHE_FILE_NAME
    if not cache_path.exists():
        return {}

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read cache at %s: %s", cache_path, e)
        return {}

    if data.get("version") != CACHE_VERSION or data.get("model") != model:
        logger.info("Cache invalidated (version/model mismatch)")
        return {}

    return data.get("directories", {})


def save_cache(repo_path: str, model: str, directories: dict[str, dict]) -> str:
    """Save directory summaries to repo's .scd_cache directory. Returns cache file path."""
    cache_dir = Path(repo_path) / CACHE_DIR_NAME
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / CACHE_FILE_NAME

    data = {
        "version": CACHE_VERSION,
        "model": model,
        "directories": directories,
    }
    cache_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(cache_path)
