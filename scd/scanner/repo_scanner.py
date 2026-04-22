from __future__ import annotations

import os
from pathlib import Path

from scd.config import SOURCE_EXTENSIONS, ScdConfig
from scd.models import DirInfo, FileInfo, RepoScanResult
from scd.scanner.ignore_rules import IgnoreRules


EXTENSION_TO_LANG: dict[str, str] = {
    ".py": "python", ".pyw": "python",
    ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".c": "c",
    ".h": "c/cpp", ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala",
    ".vue": "vue",
    ".svelte": "svelte",
}


def _detect_language(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return EXTENSION_TO_LANG.get(ext, ext.lstrip("."))


def _count_lines(content: str) -> int:
    return content.count("\n") + (1 if content and not content.endswith("\n") else 0)


def scan_repo(repo_path: str, config: ScdConfig) -> RepoScanResult:
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")

    rules = IgnoreRules(root)
    result = RepoScanResult(root_path=str(root))

    allowed_exts = SOURCE_EXTENSIONS
    if config.lang_filter:
        ext_map_rev: dict[str, set[str]] = {}
        for ext, lang in EXTENSION_TO_LANG.items():
            ext_map_rev.setdefault(lang, set()).add(ext)
        allowed_exts = set()
        for lang_name in config.lang_filter:
            lang_lower = lang_name.lower()
            if lang_lower in ext_map_rev:
                allowed_exts.update(ext_map_rev[lang_lower])
            else:
                for ext in SOURCE_EXTENSIONS:
                    if ext.lstrip(".") == lang_lower:
                        allowed_exts.add(ext)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if not rules.should_ignore_dir(d) and not d.startswith(".")
        ]

        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""

        dir_info = DirInfo(path=rel_dir)
        has_source = False

        for fname in sorted(filenames):
            if rules.should_ignore_file(fname):
                continue
            ext = Path(fname).suffix.lower()
            if ext not in allowed_exts:
                continue

            full_path = os.path.join(dirpath, fname)
            rel_file = os.path.relpath(full_path, root)

            try:
                content = Path(full_path).read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue

            line_count = _count_lines(content)
            if line_count > config.max_file_lines:
                continue

            file_info = FileInfo(
                path=rel_file,
                language=_detect_language(fname),
                line_count=line_count,
            )
            dir_info.files.append(file_info)
            result.file_contents[rel_file] = content
            has_source = True

        if has_source:
            result.dirs[rel_dir] = dir_info

    return result
