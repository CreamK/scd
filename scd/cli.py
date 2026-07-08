from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console

from scd.config import ScdConfig, load_env_file
from scd.pipeline.orchestrator import run_pipeline

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
def main() -> None:
    """SCD - Similar Code Detector: AI-powered code similarity detection."""
    pass


@main.command()
@click.argument("repo_a", type=click.Path(exists=True))
@click.argument("repo_b", type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="Markdown report file path (JSON report uses the same stem).")
@click.option("--output-dir", default=None, help="Output directory for all artifacts (default: <repo_a>_<repo_b>_output).")
@click.option("-r", "--rps", default=3.0, type=float, help="Max requests per second to the AI API.")
@click.option("-t", "--threshold", default=80, type=int, help="Minimum composite similarity score (0-100, default 80).")
@click.option("-m", "--model", default="gpt-4o-mini", help="LLM model name (OpenAI-compatible).")
@click.option("--api-key", envvar="OPENAI_API_KEY", default=None, help="API key (or set OPENAI_API_KEY env var).")
@click.option("--base-url", envvar="OPENAI_BASE_URL", default=None, help="OpenAI-compatible endpoint base URL, e.g. https://your-gateway/v1 (or set OPENAI_BASE_URL env var).")
@click.option("--lang", default=None, help="Comma-separated language filter (e.g. py,ts).")
@click.option("--shallow", is_flag=True, help="Only do directory-level matching (no function comparison).")
@click.option("--match-batch-size", default=40, type=int, help="Max directories per side per Phase 2b AI call (default 40).")
@click.option(
    "--dir-match-depth",
    type=click.Choice(["deepest", "highest"]),
    default=None,
    help="Directory overlap strategy before function comparison: deepest or highest (default deepest).",
)
@click.option("--max-in-flight", envvar="SCD_MAX_IN_FLIGHT", default=8, type=int, help="Hard cap on concurrent LLM requests (default 8; or set SCD_MAX_IN_FLIGHT env var).")
@click.option("--context-window", default=128_000, type=int, help="Model context window in tokens; oversized file pairs are chunked to fit (default 128000).")
@click.option("--json-mode/--no-json-mode", default=None, help="Force response_format=json_object on 2b/3 (auto-downgrades if endpoint rejects).")
@click.option("--parallel-tool-calls/--no-parallel-tool-calls", default=None, help="Allow parallel tool_calls in 2a (auto-downgrades if endpoint rejects).")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def compare(
    repo_a: str,
    repo_b: str,
    output: str | None,
    output_dir: str | None,
    rps: float,
    threshold: int,
    model: str,
    api_key: str | None,
    base_url: str | None,
    lang: str | None,
    shallow: bool,
    match_batch_size: int,
    dir_match_depth: str | None,
    max_in_flight: int,
    context_window: int,
    json_mode: bool | None,
    parallel_tool_calls: bool | None,
    verbose: bool,
) -> None:
    """Compare two repositories for code similarity."""
    _setup_logging(verbose)

    env = load_env_file()
    if not api_key:
        api_key = env.get("OPENAI_API_KEY") or env.get("ANTHROPIC_API_KEY")
    if not base_url:
        base_url = env.get("OPENAI_BASE_URL") or env.get("ANTHROPIC_BASE_URL")
    if model == "gpt-4o-mini":
        env_model = env.get("OPENAI_MODEL") or env.get("ANTHROPIC_MODEL")
        if env_model:
            model = env_model
    if rps == 3.0 and env.get("RPS"):
        rps = float(env["RPS"])
    if match_batch_size == 40 and env.get("MATCH_BATCH_SIZE"):
        match_batch_size = int(env["MATCH_BATCH_SIZE"])
    if max_in_flight == 8 and env.get("MAX_IN_FLIGHT"):
        max_in_flight = int(env["MAX_IN_FLIGHT"])
    if context_window == 128_000 and env.get("CONTEXT_WINDOW"):
        context_window = int(env["CONTEXT_WINDOW"])

    def _parse_bool(raw: str | None) -> bool | None:
        if raw is None:
            return None
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    if json_mode is None:
        json_mode = _parse_bool(env.get("USE_JSON_MODE")) or False
    if parallel_tool_calls is None:
        parallel_tool_calls = _parse_bool(env.get("PARALLEL_TOOL_CALLS")) or False

    dir_confidence = (env.get("DIR_CONFIDENCE") or "medium").strip().lower()
    if dir_confidence not in {"high", "medium", "low"}:
        console.print(
            f"[yellow]Warning:[/] invalid DIR_CONFIDENCE={dir_confidence!r}, "
            "expected high|medium|low; falling back to 'medium'."
        )
        dir_confidence = "medium"

    if dir_match_depth is None:
        dir_match_depth = (env.get("DIR_MATCH_DEPTH") or "deepest").strip().lower()
    if dir_match_depth not in {"deepest", "highest"}:
        console.print(
            f"[yellow]Warning:[/] invalid DIR_MATCH_DEPTH={dir_match_depth!r}, "
            "expected deepest|highest; falling back to 'deepest'."
        )
        dir_match_depth = "deepest"

    lang_filter = set()
    if lang:
        lang_filter = {l.strip().lower() for l in lang.split(",")}

    config = ScdConfig(
        api_key=api_key,
        base_url=base_url,
        rps=rps,
        similarity_threshold=threshold,
        model=model,
        output_path=output,
        output_dir=output_dir,
        lang_filter=lang_filter,
        shallow=shallow,
        match_batch_size=match_batch_size,
        max_in_flight=max_in_flight,
        context_window=context_window,
        use_json_mode=json_mode,
        parallel_tool_calls=parallel_tool_calls,
        dir_confidence=dir_confidence,
        dir_match_depth=dir_match_depth,
    )

    try:
        asyncio.run(run_pipeline(repo_a, repo_b, config))
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
