from dataclasses import dataclass, field
from pathlib import Path

ENV_FILE_NAME = ".scd.env"


def load_env_file() -> dict[str, str]:
    """Load key=value pairs from .scd.env in project root."""
    env_path = Path(__file__).resolve().parent.parent / ENV_FILE_NAME
    if not env_path.exists():
        return {}
    result: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip()
    return result


SOURCE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".go", ".java", ".rs", ".cpp", ".c", ".h", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".vue", ".svelte",
}

DEFAULT_IGNORE_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "vendor", "third_party", "3rdparty",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", "out", "target", "bin", "obj",
    ".next", ".nuxt", ".output",
    "venv", ".venv", "env", ".env",
    ".idea", ".vscode", ".cursor",
    "coverage", ".nyc_output", "htmlcov",
}

DEFAULT_IGNORE_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "go.sum",
    ".DS_Store", "Thumbs.db",
}


@dataclass
class ScdConfig:
    api_key: str | None = None
    base_url: str | None = None
    concurrency: int = 10
    similarity_threshold: int = 5
    max_file_lines: int = 10000
    model: str = "claude-sonnet-4-20250514"
    output_format: str = "markdown"
    output_path: str | None = None
    output_dir: str = "output"
    lang_filter: set[str] = field(default_factory=set)
    shallow: bool = False
