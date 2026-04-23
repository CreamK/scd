"""In-memory read-only filesystem view for a directory subtree.

Used by the tool-driven directory summarizer: the model receives `list_dir`
and `read_file` tools scoped to a single subtree of the scanned repo, and the
summarizer tracks which files were actually read to enforce a file-coverage
threshold before accepting the final JSON.
"""

from __future__ import annotations

import os
from typing import Any

from scd.models import FileInfo, RepoScanResult

READ_FILE_DEFAULT_LIMIT = 500


def _normalize(path: str) -> str | None:
    """Normalize a user-supplied relative path. Return None if it escapes."""
    if path is None:
        return ""
    p = str(path).strip()
    if p in ("", "."):
        return ""
    if os.path.isabs(p) or p.startswith("~"):
        return None
    norm = os.path.normpath(p).replace(os.sep, "/")
    if norm.startswith("..") or norm == "..":
        return None
    if norm == ".":
        return ""
    return norm.lstrip("./")


class SubtreeFS:
    """Read-only view over the subtree rooted at ``root_rel`` of ``repo``.

    - All tool paths are resolved relative to ``root_rel``.
    - ``read_paths`` records the set of files first-read by the model, used
      for coverage accounting.
    """

    def __init__(self, root_rel: str, repo: RepoScanResult) -> None:
        self._repo = repo
        self.root_rel = root_rel.strip("/").rstrip("/") if root_rel else ""

        all_files: list[FileInfo] = []
        all_dirs: list[str] = []
        prefix = f"{self.root_rel}/" if self.root_rel else ""
        for dir_path, dir_info in repo.dirs.items():
            if self.root_rel:
                if dir_path != self.root_rel and not dir_path.startswith(prefix):
                    continue
            all_dirs.append(dir_path)
            for f in dir_info.files:
                all_files.append(f)
        self.all_files: list[FileInfo] = sorted(all_files, key=lambda f: f.path)
        self.all_dirs: list[str] = sorted(all_dirs)
        self.total_files: int = len(self.all_files)
        self.total_lines: int = sum(f.line_count for f in self.all_files)
        self.read_paths: set[str] = set()

    @property
    def coverage(self) -> float:
        if self.total_files == 0:
            return 1.0
        return len(self.read_paths) / self.total_files

    def unread_files_ranked(self) -> list[FileInfo]:
        """Return files that have not been read yet, largest line_count first."""
        return sorted(
            [f for f in self.all_files if f.path not in self.read_paths],
            key=lambda f: (-f.line_count, f.path),
        )

    def _resolve(self, path: str) -> str | None:
        """Resolve a tool-supplied path to an absolute repo-relative path.

        Returns None if the path escapes the subtree.
        """
        rel = _normalize(path)
        if rel is None:
            return None
        if not self.root_rel:
            return rel
        if not rel:
            return self.root_rel
        return f"{self.root_rel}/{rel}"

    def list_dir(self, path: str = "") -> dict[str, Any]:
        target = self._resolve(path)
        if target is None:
            return {"error": "path outside subtree"}
        if target not in self._repo.dirs:
            return {"error": f"directory not found: {path or '.'}"}
        dir_info = self._repo.dirs[target]

        child_dirs: list[str] = []
        prefix = f"{target}/" if target else ""
        for d in self._repo.dirs:
            if not d or d == target:
                continue
            if prefix and not d.startswith(prefix):
                continue
            if not prefix and "/" in d:
                continue
            remainder = d[len(prefix):] if prefix else d
            if "/" in remainder:
                continue
            child_dirs.append(self._display_path(d))

        files_payload = [
            {
                "path": self._display_path(f.path),
                "language": f.language,
                "line_count": f.line_count,
            }
            for f in sorted(dir_info.files, key=lambda x: x.path)
        ]
        return {
            "path": self._display_path(target),
            "dirs": sorted(child_dirs),
            "files": files_payload,
        }

    def read_file(
        self,
        path: str,
        offset: int = 0,
        limit: int = READ_FILE_DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        target = self._resolve(path)
        if target is None:
            return {"error": "path outside subtree"}
        content = self._repo.file_contents.get(target)
        if content is None:
            return {"error": f"file not found: {path}"}
        try:
            offset = int(offset) if offset is not None else 0
            limit = int(limit) if limit is not None else READ_FILE_DEFAULT_LIMIT
        except (TypeError, ValueError):
            return {"error": "offset and limit must be integers"}
        if offset < 0:
            offset = 0
        if limit <= 0:
            limit = READ_FILE_DEFAULT_LIMIT

        lines = content.splitlines()
        total_lines = len(lines)
        chunk = lines[offset : offset + limit]
        truncated = (offset + limit) < total_lines

        if offset == 0:
            self.read_paths.add(target)

        return {
            "path": self._display_path(target),
            "offset": offset,
            "returned_lines": len(chunk),
            "total_lines": total_lines,
            "truncated": truncated,
            "content": "\n".join(chunk),
        }

    def _display_path(self, abs_rel: str) -> str:
        """Render an absolute repo-relative path as a subtree-relative path."""
        if not self.root_rel:
            return abs_rel
        if abs_rel == self.root_rel:
            return "."
        prefix = f"{self.root_rel}/"
        if abs_rel.startswith(prefix):
            return abs_rel[len(prefix):]
        return abs_rel


DIR_SUMMARY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_dir",
        "description": (
            "List immediate subdirectories and source files of a directory "
            "inside the current subtree. Paths are relative to the subtree "
            "root. Use '.' or '' for the subtree root."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the subtree root.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a source file inside the current subtree. Returns at most "
            f"{READ_FILE_DEFAULT_LIMIT} lines per call; use offset to paginate "
            "if 'truncated' is true. Reading a file contributes to coverage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the subtree root.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Zero-based line offset to start reading from.",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of lines to return. "
                        f"Defaults to {READ_FILE_DEFAULT_LIMIT}."
                    ),
                    "default": READ_FILE_DEFAULT_LIMIT,
                },
            },
            "required": ["path"],
        },
    },
]
