from __future__ import annotations

from scd.models import ScdReport, SimilarFunction, SimilarityLevel


def _level_badge(level: SimilarityLevel) -> str:
    return {
        SimilarityLevel.HIGH: "🔴 高",
        SimilarityLevel.MEDIUM: "🟡 中",
        SimilarityLevel.LOW: "🟢 低",
        SimilarityLevel.VERY_LOW: "⚪ 极低",
    }.get(level, level.value)


def _level_label(level: SimilarityLevel) -> str:
    return {
        SimilarityLevel.HIGH: "高 (>60%)",
        SimilarityLevel.MEDIUM: "中 (40-60%)",
        SimilarityLevel.LOW: "低 (20-40%)",
        SimilarityLevel.VERY_LOW: "极低 (<20%)",
    }.get(level, level.value)


def render_markdown(report: ScdReport) -> str:
    lines: list[str] = []

    lines.append("# SCD - Code Similarity Report\n")

    # --- Overview ---
    lines.append("## Overview\n")
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
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
    high = [f for f in all_funcs if f.composite_score > 60]
    medium = [f for f in all_funcs if 40 <= f.composite_score <= 60]
    low = [f for f in all_funcs if 20 <= f.composite_score < 40]
    very_low = [f for f in all_funcs if f.composite_score < 20]

    lines.append("## Similarity Distribution\n")
    lines.append("| Level | Count |")
    lines.append("|-------|-------|")
    lines.append(f"| 🔴 高 (>60%) | {len(high)} |")
    lines.append(f"| 🟡 中 (40-60%) | {len(medium)} |")
    lines.append(f"| 🟢 低 (20-40%) | {len(low)} |")
    lines.append(f"| ⚪ 极低 (<20%) | {len(very_low)} |")
    lines.append("")

    # --- Directory Match Summary ---
    if report.dir_match_result:
        dm = report.dir_match_result
        lines.append("## Directory Matches\n")
        for m in dm.matched_dirs:
            lines.append(f"- `{m.dir_a}` ↔ `{m.dir_b}` ({m.confidence}) — {m.reason}")
        lines.append("")

    # --- Detailed Results ---
    sorted_funcs = sorted(all_funcs, key=lambda f: -f.composite_score)

    lines.append("## Similar Functions (sorted by composite score)\n")
    for sf in sorted_funcs:
        _render_function_pair(lines, sf)

    return "\n".join(lines)


def _render_function_pair(lines: list[str], sf: SimilarFunction) -> None:
    badge = _level_badge(sf.similarity_level)
    lines.append(f"### {badge} — `{sf.func_a.name}` ↔ `{sf.func_b.name}` (Composite: {sf.composite_score}%)\n")
    lines.append(f"- **Level:** {_level_label(sf.similarity_level)}")
    lines.append(f"- **File A:** `{sf.func_a.file}` (lines {sf.func_a.line_start}-{sf.func_a.line_end})")
    lines.append(f"- **File B:** `{sf.func_b.file}` (lines {sf.func_b.line_start}-{sf.func_b.line_end})")
    lines.append("")
    s = sf.scores
    lines.append("| Dimension | Weight | Score |")
    lines.append("|-----------|--------|-------|")
    lines.append(f"| Data Structure | 25% | {s.data_structure}% |")
    lines.append(f"| Function Signature | 25% | {s.function_signature}% |")
    lines.append(f"| Algorithm Logic | 25% | {s.algorithm_logic}% |")
    lines.append(f"| Naming Convention | 15% | {s.naming_convention}% |")
    lines.append(f"| Protocol Conformance | 10% | {s.protocol_conformance}% |")
    lines.append("")
    lines.append(f"- **Analysis:** {sf.analysis}")
    lines.append("")
