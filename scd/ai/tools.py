"""Tool definitions and handlers for Claude agent mode."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from scd.config import SOURCE_EXTENSIONS
from scd.scanner.ignore_rules import IgnoreRules


@dataclass
class ExplorationTracker:
    """Records which directories/files the AI explores during agent loops."""

    listed_dirs: list[dict] = field(default_factory=list)
    read_files: list[dict] = field(default_factory=list)
    _call_seq: int = field(default=0, repr=False)

    def _next_seq(self) -> int:
        self._call_seq += 1
        return self._call_seq

    def record_list_dir(self, path: str, relative: str, repo_label: str) -> None:
        self.listed_dirs.append({
            "seq": self._next_seq(),
            "path": path,
            "relative": relative,
            "repo": repo_label,
        })

    def record_read_file(self, path: str, relative: str, repo_label: str, max_lines: int) -> None:
        self.read_files.append({
            "seq": self._next_seq(),
            "path": path,
            "relative": relative,
            "repo": repo_label,
            "max_lines": max_lines,
        })

    def to_dict(self) -> dict:
        return {
            "total_list_directory_calls": len(self.listed_dirs),
            "total_read_file_calls": len(self.read_files),
            "listed_directories": self.listed_dirs,
            "read_files": self.read_files,
        }

    def to_tree_text(self) -> str:
        """Render exploration history as readable file trees, one per repo."""
        repo_dirs: dict[str, set[str]] = {}
        repo_files: dict[str, dict[str, int]] = {}

        for entry in self.listed_dirs:
            repo = entry["repo"]
            rel = entry["relative"]
            repo_dirs.setdefault(repo, set()).add(rel)

        for entry in self.read_files:
            repo = entry["repo"]
            rel = entry["relative"]
            repo_files.setdefault(repo, {})[rel] = entry.get("max_lines", 50)

        lines: list[str] = [
            f"Exploration Stats: {len(self.listed_dirs)} dir listings, "
            f"{len(self.read_files)} file reads",
            "",
        ]

        for repo in sorted(set(list(repo_dirs.keys()) + list(repo_files.keys()))):
            lines.append(f"Repo {repo}:")
            tree = _build_tree(repo_dirs.get(repo, set()), repo_files.get(repo, {}))
            lines.append(tree)
            lines.append("")

        return "\n".join(lines).rstrip()


def _build_tree(dirs: set[str], files: dict[str, int]) -> str:
    """Build an ASCII tree from directory paths and read file paths."""
    tree: dict = {}

    for d in dirs:
        parts = d.split("/") if d else []
        node = tree
        for part in parts:
            node = node.setdefault(part + "/", {})

    for f, max_lines in files.items():
        parts = f.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part + "/", {})
        node[f"* {parts[-1]} [read:{max_lines}L]"] = None

    return _render_tree(tree, "")


def _render_tree(node: dict, prefix: str) -> str:
    lines: list[str] = []
    entries = sorted(node.keys(), key=lambda k: (not k.endswith("/"), k.lower()))
    for i, key in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{key}")
        child = node[key]
        if child is not None and isinstance(child, dict) and child:
            extension = "    " if is_last else "│   "
            lines.append(_render_tree(child, prefix + extension))
    return "\n".join(lines)

TOOL_LIST_DIR = {
    "name": "list_directory",
    "description": (
        "List the contents of a directory. Returns subdirectories and source code files. "
        "Use this to explore repository structure. Start from the repo root and drill down."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the directory to list.",
            },
        },
        "required": ["path"],
    },
}

TOOL_READ_FILE = {
    "name": "read_file",
    "description": (
        "Read the first N lines of a source code file. "
        "Use this to peek at file contents when you need more context to judge similarity."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the file to read.",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to read (default 50).",
                "default": 50,
            },
        },
        "required": ["path"],
    },
}

ALL_TOOLS = [TOOL_LIST_DIR, TOOL_READ_FILE]


def create_tool_handler(
    allowed_roots: list[str],
    tracker: ExplorationTracker | None = None,
):
    """Create a tool handler that only allows access within the given root paths."""
    resolved_roots = [str(Path(r).resolve()) for r in allowed_roots]

    def _is_allowed(path: str) -> bool:
        resolved = str(Path(path).resolve())
        return any(resolved.startswith(root) for root in resolved_roots)

    def _resolve_repo_label(path: str) -> tuple[str, str]:
        """Return (repo_label, relative_path) for a given absolute path."""
        resolved = str(Path(path).resolve())
        for i, root in enumerate(resolved_roots):
            if resolved.startswith(root):
                label = chr(ord("A") + i)
                rel = os.path.relpath(resolved, root)
                if rel == ".":
                    rel = ""
                return label, rel
        return "?", path

    ignore_caches: dict[str, IgnoreRules] = {}

    def _get_ignore_rules(path: str) -> IgnoreRules:
        for root in resolved_roots:
            if str(Path(path).resolve()).startswith(root):
                if root not in ignore_caches:
                    ignore_caches[root] = IgnoreRules(Path(root))
                return ignore_caches[root]
        return IgnoreRules(Path(path))

    async def handler(tool_name: str, tool_input: dict) -> str:
        if tool_name == "list_directory":
            result = _handle_list_dir(tool_input, _is_allowed, _get_ignore_rules)
            if tracker and not result.startswith("Error"):
                label, rel = _resolve_repo_label(tool_input.get("path", ""))
                tracker.record_list_dir(tool_input.get("path", ""), rel, label)
            return result
        elif tool_name == "read_file":
            result = _handle_read_file(tool_input, _is_allowed)
            if tracker and not result.startswith("Error"):
                label, rel = _resolve_repo_label(tool_input.get("path", ""))
                tracker.record_read_file(
                    tool_input.get("path", ""), rel, label,
                    tool_input.get("max_lines", 50),
                )
            return result
        else:
            return f"Unknown tool: {tool_name}"

    return handler


def _handle_list_dir(
    tool_input: dict,
    is_allowed: callable,
    get_ignore_rules: callable,
) -> str:
    path = tool_input.get("path", "")
    if not is_allowed(path):
        return "Error: Access denied — path is outside allowed repositories."

    p = Path(path)
    if not p.is_dir():
        return f"Error: Not a directory: {path}"

    rules = get_ignore_rules(path)

    dirs: list[str] = []
    files: list[str] = []

    try:
        for entry in sorted(p.iterdir()):
            name = entry.name
            if entry.is_dir():
                if not rules.should_ignore_dir(name) and not name.startswith("."):
                    dirs.append(f"  {name}/")
            elif entry.is_file():
                ext = entry.suffix.lower()
                if ext in SOURCE_EXTENSIONS and not rules.should_ignore_file(name):
                    files.append(f"  {name}")
    except PermissionError:
        return f"Error: Permission denied: {path}"

    parts: list[str] = []
    if dirs:
        parts.append("Directories:\n" + "\n".join(dirs))
    if files:
        parts.append("Files:\n" + "\n".join(files))
    if not parts:
        return "Empty directory (no source code files or subdirectories)."

    return "\n".join(parts)


def _handle_read_file(tool_input: dict, is_allowed: callable) -> str:
    path = tool_input.get("path", "")
    max_lines = tool_input.get("max_lines", 50)

    if not is_allowed(path):
        return "Error: Access denied — path is outside allowed repositories."

    p = Path(path)
    if not p.is_file():
        return f"Error: Not a file: {path}"

    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError) as e:
        return f"Error reading file: {e}"

    lines = content.splitlines()
    total = len(lines)
    truncated = lines[:max_lines]
    result = "\n".join(truncated)

    if total > max_lines:
        result += f"\n\n... ({total - max_lines} more lines, {total} total)"

    return result
