from __future__ import annotations

from scd.models import ScdReport, SimilarFunction, SimilarityType


def _score_badge(score: int) -> str:
    if score >= 8:
        return "🔴 HIGH"
    elif score >= 5:
        return "🟡 MEDIUM"
    return "🟢 LOW"


def _type_label(t: SimilarityType) -> str:
    return {
        SimilarityType.COPY: "Copy",
        SimilarityType.LOGIC_IDENTICAL: "Logic Identical",
        SimilarityType.PARTIAL: "Partial",
        SimilarityType.UNRELATED: "Unrelated",
    }.get(t, t.value)


def render_markdown(report: ScdReport) -> str:
    lines: list[str] = []

    lines.append("# SCD - Code Similarity Report\n")

    # --- Overview ---
    lines.append("## Overview\n")
    lines.append(f"| Item | Value |")
    lines.append(f"|------|-------|")
    lines.append(f"| Repo A | `{report.repo_a_path}` |")
    lines.append(f"| Repo B | `{report.repo_b_path}` |")
    lines.append(f"| Files in A | {report.repo_a_files} |")
    lines.append(f"| Files in B | {report.repo_b_files} |")
    lines.append(f"| Total AI Calls | {report.total_ai_calls} |")

    all_funcs = report.all_similar_functions
    lines.append(f"| Similar Function Pairs | {len(all_funcs)} |")
    lines.append("")

    if not all_funcs:
        lines.append("**No similar functions found.**\n")
        return "\n".join(lines)

    # --- Score Distribution ---
    high = [f for f in all_funcs if f.similarity_score >= 8]
    medium = [f for f in all_funcs if 5 <= f.similarity_score < 8]
    low = [f for f in all_funcs if f.similarity_score < 5]

    lines.append("## Similarity Distribution\n")
    lines.append(f"| Level | Count |")
    lines.append(f"|-------|-------|")
    lines.append(f"| 🔴 High (8-10) | {len(high)} |")
    lines.append(f"| 🟡 Medium (5-7) | {len(medium)} |")
    lines.append(f"| 🟢 Low (3-4) | {len(low)} |")
    lines.append("")

    # --- Directory Match Summary ---
    if report.dir_match_result:
        dm = report.dir_match_result
        lines.append("## Directory Matches\n")
        for m in dm.matched_dirs:
            lines.append(f"- `{m.dir_a}` ↔ `{m.dir_b}` ({m.confidence}) — {m.reason}")
        if dm.orphan_dirs_a:
            lines.append(f"\n**Unmatched in A:** {', '.join(f'`{d}`' for d in dm.orphan_dirs_a)}")
        if dm.orphan_dirs_b:
            lines.append(f"\n**Unmatched in B:** {', '.join(f'`{d}`' for d in dm.orphan_dirs_b)}")
        lines.append("")

    # --- Detailed Results ---
    sorted_funcs = sorted(all_funcs, key=lambda f: -f.similarity_score)

    lines.append("## Similar Functions (sorted by score)\n")
    for sf in sorted_funcs:
        _render_function_pair(lines, sf)

    return "\n".join(lines)


def _render_function_pair(lines: list[str], sf: SimilarFunction) -> None:
    badge = _score_badge(sf.similarity_score)
    lines.append(f"### {badge} — `{sf.func_a.name}` ↔ `{sf.func_b.name}` (Score: {sf.similarity_score}/10)\n")
    lines.append(f"- **Type:** {_type_label(sf.similarity_type)}")
    lines.append(f"- **File A:** `{sf.func_a.file}` (lines {sf.func_a.line_start}-{sf.func_a.line_end})")
    lines.append(f"- **File B:** `{sf.func_b.file}` (lines {sf.func_b.line_start}-{sf.func_b.line_end})")
    lines.append(f"- **Analysis:** {sf.analysis}")
    lines.append("")
