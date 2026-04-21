from __future__ import annotations

import fnmatch
from pathlib import Path

from scd.config import DEFAULT_IGNORE_DIRS, DEFAULT_IGNORE_FILES


class IgnoreRules:
    """Manages ignore rules for scanning, combining defaults with .scdignore."""

    def __init__(self, repo_root: Path) -> None:
        self._ignore_dirs = set(DEFAULT_IGNORE_DIRS)
        self._ignore_files = set(DEFAULT_IGNORE_FILES)
        self._ignore_patterns: list[str] = []
        self._load_scdignore(repo_root)

    def _load_scdignore(self, repo_root: Path) -> None:
        ignore_file = repo_root / ".scdignore"
        if not ignore_file.exists():
            return
        for line in ignore_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith("/"):
                self._ignore_dirs.add(line.rstrip("/"))
            else:
                self._ignore_patterns.append(line)

    def should_ignore_dir(self, dir_name: str) -> bool:
        return dir_name in self._ignore_dirs

    def should_ignore_file(self, file_name: str) -> bool:
        if file_name in self._ignore_files:
            return True
        return any(fnmatch.fnmatch(file_name, p) for p in self._ignore_patterns)
