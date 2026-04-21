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
@click.option("-o", "--output", default=None, help="Output report file path (overrides --output-dir default).")
@click.option("--output-dir", default="output", help="Output directory for all artifacts (default: output).")
@click.option("-f", "--format", "fmt", type=click.Choice(["markdown", "json"]), default="markdown", help="Output format.")
@click.option("-c", "--concurrency", default=10, type=int, help="Max concurrent AI calls.")
@click.option("-t", "--threshold", default=20, type=int, help="Minimum composite similarity score (0-100, default 20).")
@click.option("-m", "--model", default="claude-sonnet-4-20250514", help="Claude model to use.")
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY env var).")
@click.option("--base-url", envvar="ANTHROPIC_BASE_URL", default=None, help="Anthropic API base URL (or set ANTHROPIC_BASE_URL env var).")
@click.option("--lang", default=None, help="Comma-separated language filter (e.g. py,ts).")
@click.option("--shallow", is_flag=True, help="Only do directory-level matching (no function comparison).")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def compare(
    repo_a: str,
    repo_b: str,
    output: str | None,
    output_dir: str,
    fmt: str,
    concurrency: int,
    threshold: int,
    model: str,
    api_key: str | None,
    base_url: str | None,
    lang: str | None,
    shallow: bool,
    verbose: bool,
) -> None:
    """Compare two repositories for code similarity."""
    _setup_logging(verbose)

    env = load_env_file()
    if not api_key:
        api_key = env.get("ANTHROPIC_API_KEY")
    if not base_url:
        base_url = env.get("ANTHROPIC_BASE_URL")
    if model == "claude-sonnet-4-20250514" and env.get("ANTHROPIC_MODEL"):
        model = env["ANTHROPIC_MODEL"]

    lang_filter = set()
    if lang:
        lang_filter = {l.strip().lower() for l in lang.split(",")}

    config = ScdConfig(
        api_key=api_key,
        base_url=base_url,
        concurrency=concurrency,
        similarity_threshold=threshold,
        model=model,
        output_format=fmt,
        output_path=output,
        output_dir=output_dir,
        lang_filter=lang_filter,
        shallow=shallow,
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
