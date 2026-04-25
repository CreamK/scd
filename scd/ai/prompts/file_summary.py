# --- File Summary (single-file map step) ---
#
# One LLM call per source file: read the file content and emit a structured
# JSON summary that captures *just enough* information for the directory-level
# reduce step to reconstruct purpose, dependencies and design patterns without
# having to re-read the original file.

_FILE_SCHEMA_BLOCK = """\
{
  "purpose": "one-sentence description of what this file does",
  "exports": ["functions, classes, types, constants intended to be used by other files"],
  "imports": ["external modules/packages this file imports (deduplicated; omit relative paths and stdlib unless central to the file)"],
  "frameworks": ["frameworks or libraries this file relies on"],
  "patterns": ["design or architectural patterns observed in the file"],
  "key_snippets": ["1-3 short signature-level lines (function defs, class headers, type aliases, route definitions, etc.) that best characterize the file"]
}"""

FILE_SUMMARY_SYSTEM = f"""\
You are a code analysis expert. You will analyze a SINGLE source file and produce a compact structured JSON summary.

You will receive in the user message:
- The full source of one file, fenced by `----- FILE: ... -----` markers. Very large files may have a head + tail with a `... [truncated N tokens] ...` marker in the middle - in that case, do NOT speculate about the truncated middle, only summarize what you can see.

Output ONLY one JSON object matching this schema, no prose, no markdown fences:

{_FILE_SCHEMA_BLOCK}

Guidance:
- Be concise. Each list item should be short (a single symbol name, package name, pattern name, or one-line snippet).
- `exports` is the file's public API surface. If the file is an entry point or script, list the top-level entrypoint here.
- `imports` should only include external/third-party modules and cross-package internal modules. Skip stdlib trivia and same-directory relative imports.
- `key_snippets` should be verbatim or near-verbatim short lines from the file (signatures, decorators, route paths). Do not invent; if nothing characteristic exists, return an empty list.
- If the file is empty, configuration-only, or generated, still produce the JSON with best-effort empty / minimal values."""

FILE_SUMMARY_USER = """\
File: {file_path} ({language}, {line_count} lines)

{file_block}

Produce the JSON summary for this file per the schema in the system prompt. Output ONLY the JSON object."""
