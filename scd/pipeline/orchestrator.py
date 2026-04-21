from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from rich.console import Console

from scd.ai.client import ClaudeClient
from scd.config import ScdConfig
from scd.models import RepoScanResult, ScdReport
from scd.pipeline.directory_matcher import match_directories
from scd.pipeline.function_comparer import build_all_file_pairs, compare_file_pairs, deduplicate_results
from scd.pipeline.orphan_handler import handle_orphan_dirs
from scd.reporter.reporter import save_report
from scd.scanner.repo_scanner import scan_repo

logger = logging.getLogger(__name__)
console = Console()


def _write_exploration_log(output_dir: str, exploration_log: dict, exploration_tree: str = "") -> str:
    """Write exploration log to output directory, return the tree file path."""
    json_path = os.path.join(output_dir, "exploration_log.json")
    Path(json_path).write_text(
        json.dumps(exploration_log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    tree_path = os.path.join(output_dir, "exploration_log.txt")
    if exploration_tree:
        Path(tree_path).write_text(exploration_tree + "\n", encoding="utf-8")
        return tree_path
    return json_path


def _write_compared_pairs(output_dir: str, file_pairs: list[tuple[str, str]]) -> str:
    """Write all file pairs to compare to a text file, one pair per line."""
    path = os.path.join(output_dir, "compared_pairs.txt")
    lines = [f"{fa} -> {fb}" for fa, fb in file_pairs]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def run_pipeline(repo_a_path: str, repo_b_path: str, config: ScdConfig) -> ScdReport:
    """Run the full SCD comparison pipeline."""
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    client = ClaudeClient(config)

    # --- Phase 1: Scan ---
    console.print("\n[bold blue]Phase 1:[/] Scanning repositories...")
    t0 = time.monotonic()

    repo_a = scan_repo(repo_a_path, config)
    repo_b = scan_repo(repo_b_path, config)

    console.print(f"  Repo A: {repo_a.total_files} files in {len(repo_a.dirs)} directories")
    console.print(f"  Repo B: {repo_b.total_files} files in {len(repo_b.dirs)} directories")
    console.print(f"  Scan completed in {time.monotonic() - t0:.1f}s")

    report = ScdReport(
        repo_a_path=repo_a_path,
        repo_b_path=repo_b_path,
        repo_a_files=repo_a.total_files,
        repo_b_files=repo_b.total_files,
    )

    if config.shallow:
        # --- Phase 2a only: Directory matching (shallow) ---
        console.print("\n[bold blue]Phase 2a:[/] Matching directories (shallow mode)...")
        t1 = time.monotonic()
        dir_result = await match_directories(repo_a, repo_b, client)
        report.dir_match_result = dir_result
        report.total_ai_calls = client.total_calls
        console.print(f"  Found {len(dir_result.matched_dirs)} directory pairs")
        console.print(f"  Directory matching completed in {time.monotonic() - t1:.1f}s")

        if dir_result.exploration_log:
            log_path = _write_exploration_log(output_dir, dir_result.exploration_log, dir_result.exploration_tree)
            console.print(f"  Exploration log saved to [bold]{log_path}[/]")

        report_ext = "json" if config.output_format == "json" else "md"
        report_path = config.output_path or os.path.join(output_dir, f"report.{report_ext}")
        save_report(report, report_path, config.output_format)
        console.print(f"\n[bold green]Done![/] Report saved to [bold]{report_path}[/]")
        return report

    # --- Phase 2a: Directory matching ---
    console.print("\n[bold blue]Phase 2a:[/] Matching directories...")
    t1 = time.monotonic()
    dir_result = await match_directories(repo_a, repo_b, client)
    report.dir_match_result = dir_result

    for m in dir_result.matched_dirs:
        console.print(f"  [green]✓[/] {m.dir_a} <-> {m.dir_b} ({m.confidence})")
    if dir_result.orphan_dirs_a:
        console.print(f"  [yellow]Orphan dirs in A:[/] {', '.join(dir_result.orphan_dirs_a)}")
    if dir_result.orphan_dirs_b:
        console.print(f"  [yellow]Orphan dirs in B:[/] {', '.join(dir_result.orphan_dirs_b)}")
    console.print(f"  Directory matching completed in {time.monotonic() - t1:.1f}s")

    if dir_result.exploration_log:
        log_path = _write_exploration_log(output_dir, dir_result.exploration_log, dir_result.exploration_tree)
        console.print(f"  Exploration log saved to [bold]{log_path}[/]")

    # --- Phase 2b: Orphan recovery ---
    extra_matches = []
    if dir_result.orphan_dirs_a or dir_result.orphan_dirs_b:
        console.print("\n[bold blue]Phase 2b:[/] Checking orphan directories...")
        t2 = time.monotonic()
        extra_matches = await handle_orphan_dirs(
            dir_result.orphan_dirs_a, dir_result.orphan_dirs_b,
            repo_a, repo_b, client,
        )
        for m in extra_matches:
            console.print(f"  [cyan]↺[/] {m.dir_a} <-> {m.dir_b} (recovered)")
        console.print(f"  Orphan recovery completed in {time.monotonic() - t2:.1f}s")

    all_dir_matches = dir_result.matched_dirs + extra_matches

    # Build file pairs and write compared_pairs.txt
    all_file_pairs = build_all_file_pairs(all_dir_matches, repo_a, repo_b)
    pairs_path = _write_compared_pairs(output_dir, all_file_pairs)
    console.print(f"\n  File pairs determined: {len(all_file_pairs)} pairs")
    console.print(f"  Compared pairs saved to [bold]{pairs_path}[/]")

    # --- Phase 3: Function comparison ---
    console.print("\n[bold blue]Phase 3:[/] Comparing functions...")
    t3 = time.monotonic()
    raw_results = await compare_file_pairs(all_file_pairs, repo_a, repo_b, client, config)
    report.compare_results = deduplicate_results(raw_results)

    total_similar = sum(len(cr.similar_functions) for cr in report.compare_results)
    console.print(f"  Found {total_similar} similar function pairs across {len(report.compare_results)} file pairs")
    console.print(f"  Function comparison completed in {time.monotonic() - t3:.1f}s")

    report.total_ai_calls = client.total_calls

    report_ext = "json" if config.output_format == "json" else "md"
    report_path = config.output_path or os.path.join(output_dir, f"report.{report_ext}")
    save_report(report, report_path, config.output_format)

    total_time = time.monotonic() - t0
    console.print(f"\n[bold green]Done![/] Total time: {total_time:.1f}s, AI calls: {report.total_ai_calls}")
    console.print(f"  Report saved to [bold]{report_path}[/]")

    return report
