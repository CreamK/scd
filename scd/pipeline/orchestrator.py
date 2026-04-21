from __future__ import annotations

import logging
import time

from rich.console import Console

from scd.ai.client import ClaudeClient
from scd.config import ScdConfig
from scd.models import RepoScanResult, ScdReport
from scd.pipeline.directory_matcher import match_directories
from scd.pipeline.function_comparer import compare_matched_dirs, deduplicate_results
from scd.pipeline.orphan_handler import handle_orphan_dirs
from scd.scanner.repo_scanner import scan_repo

logger = logging.getLogger(__name__)
console = Console()


async def run_pipeline(repo_a_path: str, repo_b_path: str, config: ScdConfig) -> ScdReport:
    """Run the full SCD comparison pipeline."""
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
        # --- Phase 2 only: Directory matching ---
        console.print("\n[bold blue]Phase 2:[/] Matching directories (shallow mode)...")
        t1 = time.monotonic()
        dir_result = await match_directories(repo_a, repo_b, client)
        report.dir_match_result = dir_result
        report.total_ai_calls = client.total_calls
        console.print(f"  Found {len(dir_result.matched_dirs)} directory pairs")
        console.print(f"  Directory matching completed in {time.monotonic() - t1:.1f}s")
        return report

    # --- Phase 2: Directory matching ---
    console.print("\n[bold blue]Phase 2:[/] Matching directories...")
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

    # --- Phase 3a: Orphan handling ---
    extra_matches = []
    if dir_result.orphan_dirs_a or dir_result.orphan_dirs_b:
        console.print("\n[bold blue]Phase 3a:[/] Checking orphan directories...")
        t2 = time.monotonic()
        extra_matches = await handle_orphan_dirs(
            dir_result.orphan_dirs_a, dir_result.orphan_dirs_b,
            repo_a, repo_b, client,
        )
        for m in extra_matches:
            console.print(f"  [cyan]↺[/] {m.dir_a} <-> {m.dir_b} (recovered)")
        console.print(f"  Orphan check completed in {time.monotonic() - t2:.1f}s")

    all_dir_matches = dir_result.matched_dirs + extra_matches

    # --- Phase 3b: Function comparison ---
    console.print("\n[bold blue]Phase 3b:[/] Comparing functions...")
    t3 = time.monotonic()
    raw_results = await compare_matched_dirs(all_dir_matches, repo_a, repo_b, client, config)
    report.compare_results = deduplicate_results(raw_results)

    total_similar = sum(len(cr.similar_functions) for cr in report.compare_results)
    console.print(f"  Found {total_similar} similar function pairs across {len(report.compare_results)} file pairs")
    console.print(f"  Function comparison completed in {time.monotonic() - t3:.1f}s")

    report.total_ai_calls = client.total_calls

    total_time = time.monotonic() - t0
    console.print(f"\n[bold green]Done![/] Total time: {total_time:.1f}s, AI calls: {report.total_ai_calls}")

    return report
