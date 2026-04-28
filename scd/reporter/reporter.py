from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scd.models import ScdReport
from scd.reporter.markdown_template import extract_code_snippet, render_markdown


def _report_to_dict(report: ScdReport) -> dict:
    """Convert report to a JSON-serializable dict."""
    data = asdict(report)
    _add_code_snippets(data, report)
    return data


def _add_code_snippets(data: dict, report: ScdReport) -> None:
    """Attach source snippets to function locations in JSON reports."""
    for compare_result in data.get("compare_results", []):
        for similar in compare_result.get("similar_functions", []):
            func_a = similar.get("func_a", {})
            func_b = similar.get("func_b", {})
            func_a["code"] = extract_code_snippet(
                report.repo_a_path,
                func_a.get("file", ""),
                int(func_a.get("line_start", 0) or 0),
                int(func_a.get("line_end", 0) or 0),
            )
            func_b["code"] = extract_code_snippet(
                report.repo_b_path,
                func_b.get("file", ""),
                int(func_b.get("line_start", 0) or 0),
                int(func_b.get("line_end", 0) or 0),
            )


def save_json(report: ScdReport, path: str) -> None:
    data = _report_to_dict(report)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_markdown(report: ScdReport, path: str) -> None:
    md = render_markdown(report)
    Path(path).write_text(md, encoding="utf-8")


def save_report(report: ScdReport, path: str, fmt: str) -> None:
    if fmt == "json":
        save_json(report, path)
    elif fmt == "markdown":
        save_markdown(report, path)
    else:
        raise ValueError(f"Unknown format: {fmt}")
