from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scd.models import ScdReport
from scd.reporter.markdown_template import render_markdown


def _report_to_dict(report: ScdReport) -> dict:
    """Convert report to a JSON-serializable dict."""
    data = asdict(report)
    return data


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
