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
from scd.pipeline.dir_summarizer import summarize_repo
from scd.pipeline.directory_matcher import (
    compute_match_key,
    load_match_cache,
    match_directories,
    save_match_cache,
)
from scd.pipeline.function_comparer import (
    PairCache,
    build_all_file_pairs,
    compare_file_pairs,
    deduplicate_results,
)
from scd.reporter.reporter import save_report
from scd.scanner.repo_scanner import scan_repo

logger = logging.getLogger(__name__)
console = Console()


def _write_summaries(output_dir: str, summaries_a: dict[str, str], summaries_b: dict[str, str]) -> str:
    """Write directory summaries to output directory for inspection."""
    path = os.path.join(output_dir, "dir_summaries.json")
    data = {"repo_a": summaries_a, "repo_b": summaries_b}
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


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

    # --- Phase 2a: Generate directory summaries ---
    console.print("\n[bold blue]Phase 2a:[/] Generating directory summaries...")
    t1 = time.monotonic()

    summaries_a = await summarize_repo(repo_a, client, config.model)
    console.print(f"  Repo A: {len(summaries_a)} directory summaries")

    summaries_b = await summarize_repo(repo_b, client, config.model)
    console.print(f"  Repo B: {len(summaries_b)} directory summaries")

    console.print(f"  Summaries completed in {time.monotonic() - t1:.1f}s")

    summaries_path = _write_summaries(output_dir, summaries_a, summaries_b)
    console.print(f"  Summaries saved to [bold]{summaries_path}[/]")

    # --- Phase 2b: Match directories ---
    console.print("\n[bold blue]Phase 2b:[/] Matching directories...")
    t2 = time.monotonic()

    match_summaries_a = {k: v for k, v in summaries_a.items() if k != ""}
    match_summaries_b = {k: v for k, v in summaries_b.items() if k != ""}
    dropped_a = len(summaries_a) - len(match_summaries_a)
    dropped_b = len(summaries_b) - len(match_summaries_b)
    if dropped_a or dropped_b:
        logger.info(
            "Excluding root directories from matching (A=%d, B=%d dropped)",
            dropped_a, dropped_b,
        )

    match_key = compute_match_key(
        match_summaries_a, match_summaries_b, config.model, config.match_batch_size,
    )
    cached_match = load_match_cache(output_dir, match_key)
    if cached_match is not None:
        dir_result = cached_match
        console.print(
            f"  Loaded cached match ({len(dir_result.matched_dirs)} pairs)"
        )
    else:
        dir_result = await match_directories(
            repo_a, repo_b, match_summaries_a, match_summaries_b, client,
            batch_size=config.match_batch_size,
        )
        cache_path = save_match_cache(output_dir, match_key, dir_result)
        console.print(f"  Match cached to [bold]{cache_path}[/]")
    report.dir_match_result = dir_result

    for m in dir_result.matched_dirs:
        console.print(f"  [green]\u2713[/] {m.dir_a} <-> {m.dir_b} ({m.confidence})")
    console.print(f"  Matched {len(dir_result.matched_dirs)} directory pairs in {time.monotonic() - t2:.1f}s")

    if config.shallow:
        report.total_ai_calls = client.total_calls
        report_ext = "json" if config.output_format == "json" else "md"
        report_path = config.output_path or os.path.join(output_dir, f"report.{report_ext}")
        save_report(report, report_path, config.output_format)
        console.print(f"\n[bold green]Done![/] Report saved to [bold]{report_path}[/]")
        return report

    # Build file pairs and write compared_pairs.txt
    all_file_pairs = build_all_file_pairs(dir_result.matched_dirs, repo_a, repo_b)
    pairs_path = _write_compared_pairs(output_dir, all_file_pairs)
    console.print(f"\n  File pairs determined: {len(all_file_pairs)} pairs")
    console.print(f"  Compared pairs saved to [bold]{pairs_path}[/]")

    # --- Phase 3: Function comparison ---
    console.print("\n[bold blue]Phase 3:[/] Comparing functions...")
    t3 = time.monotonic()

    pair_cache = PairCache(output_dir)
    resumed = pair_cache.load()
    if resumed:
        console.print(
            f"  Resuming from checkpoint: {resumed} pair result(s) loaded from "
            f"[bold]{pair_cache.path}[/]"
        )
    else:
        console.print(f"  Checkpoint: [bold]{pair_cache.path}[/] (new)")

    raw_results = await compare_file_pairs(
        all_file_pairs, repo_a, repo_b, client, config, cache=pair_cache,
    )
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
