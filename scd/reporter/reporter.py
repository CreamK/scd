from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

from scd.models import FuncLocation, ScdReport, SimilarFunction
from scd.reporter.markdown_template import render_markdown


def _safe_source_path(repo_root: str, file_path: str) -> Path | None:
    root = Path(repo_root).resolve()
    source = (root / file_path).resolve()
    try:
        source.relative_to(root)
    except ValueError:
        return None
    return source


def _read_source_text(repo_root: str, file_path: str) -> str:
    source = _safe_source_path(repo_root, file_path)
    if source is None:
        return ""
    try:
        return source.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _line_window(lines: list[str], line_start: int, line_end: int, context: int = 20) -> tuple[int, int]:
    if not lines or line_start < 1 or line_end < line_start:
        return (1, 0)
    start = max(1, line_start - context)
    end = min(len(lines), line_end + context)
    return (start, end)


def _render_html_side(
    label: str,
    file_path: str,
    source_text: str,
    location: FuncLocation,
) -> str:
    lines = source_text.splitlines()
    start, end = _line_window(lines, location.line_start, location.line_end)
    rendered_lines: list[str] = []
    highlight_color = "#ffe4e6" if label == "huawei_file" else "#dcfce7"
    for line_no in range(start, end + 1):
        escaped_line = html.escape(lines[line_no - 1])
        line_no_html = f'<span class="scd-line-no">{line_no:>4} | </span>'
        if location.line_start <= line_no <= location.line_end:
            code_html = (
                '<span class="scd-highlight" '
                f'style="display:inline-block;width:100%;background-color:{highlight_color};">'
                f"{escaped_line}</span>"
            )
        else:
            code_html = f'<span class="scd-code">{escaped_line}</span>'
        rendered_lines.append(
            f'<span class="scd-line" style="display:block;">{line_no_html}{code_html}</span>'
        )

    side_class = "scd-side-left" if label == "huawei_file" else "scd-side-right"
    body_html = "\n".join(rendered_lines)
    return (
        f'<div class="scd-side {side_class}" style="flex:1;min-width:0;">'
        f'<div class="scd-file-label" style="font-weight:600;margin-bottom:8px;">'
        f"{html.escape(label)}: {html.escape(file_path)}</div>"
        f'<pre style="margin:0;overflow:auto;"><code>{body_html}</code></pre>'
        f'</div>'
    )


def _render_html_context(report: ScdReport, sf: SimilarFunction) -> str:
    source_a = _read_source_text(report.repo_a_path, sf.func_a.file)
    source_b = _read_source_text(report.repo_b_path, sf.func_b.file)
    return (
        '<div class="scd-html-context" style="display:flex;gap:16px;">'
        + _render_html_side("huawei_file", sf.func_a.file, source_a, sf.func_a)
        + _render_html_side("Linux_file", sf.func_b.file, source_b, sf.func_b)
        + "</div>"
    )


def _pair_hash(report: ScdReport, sf: SimilarFunction) -> str:
    source_a = _read_source_text(report.repo_a_path, sf.func_a.file)
    source_b = _read_source_text(report.repo_b_path, sf.func_b.file)
    return hashlib.sha256((source_a + source_b).encode("utf-8")).hexdigest()


def _review_json_records(report: ScdReport) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for sf in sorted(report.all_similar_functions, key=lambda f: -f.composite_score):
        records.append(
            {
                "huawei_file": sf.func_a.file,
                "Linux_file": sf.func_b.file,
                "html_context": _render_html_context(report, sf),
                "reason": sf.analysis,
                "severity": sf.similarity_level.value,
                "hash": _pair_hash(report, sf),
            }
        )
    return records


def save_json(report: ScdReport, path: str) -> None:
    data = _review_json_records(report)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_markdown(report: ScdReport, path: str) -> None:
    md = render_markdown(report)
    Path(path).write_text(md, encoding="utf-8")


def save_reports(report: ScdReport, markdown_path: str, json_path: str) -> None:
    save_markdown(report, markdown_path)
    save_json(report, json_path)


def save_report(report: ScdReport, path: str, fmt: str) -> None:
    if fmt == "json":
        save_json(report, path)
    elif fmt == "markdown":
        save_markdown(report, path)
    else:
        raise ValueError(f"Unknown format: {fmt}")
