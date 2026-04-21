from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class FileInfo:
    path: str  # relative to repo root
    language: str
    line_count: int
    size_bytes: int


@dataclass
class DirInfo:
    path: str  # relative to repo root
    files: list[FileInfo] = field(default_factory=list)
    subdirs: list[str] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def lang_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for f in self.files:
            dist[f.language] = dist.get(f.language, 0) + 1
        return dist


@dataclass
class RepoScanResult:
    root_path: str
    dirs: dict[str, DirInfo] = field(default_factory=dict)
    file_contents: dict[str, str] = field(default_factory=dict)

    @property
    def total_files(self) -> int:
        return len(self.file_contents)

    def file_paths_text(self) -> str:
        """Generate a simple list of all file paths for AI consumption."""
        return "\n".join(sorted(self.file_contents.keys()))

    def dir_paths_text(self) -> str:
        """Generate a simple list of directory paths for AI consumption."""
        return "\n".join(sorted(self.dirs.keys()))


@dataclass
class DirMatch:
    dir_a: str
    dir_b: str
    confidence: str
    reason: str


@dataclass
class DirMatchResult:
    matched_dirs: list[DirMatch] = field(default_factory=list)
    orphan_dirs_a: list[str] = field(default_factory=list)
    orphan_dirs_b: list[str] = field(default_factory=list)
    exploration_log: dict = field(default_factory=dict)


class SimilarityType(str, Enum):
    COPY = "copy"
    LOGIC_IDENTICAL = "logic_identical"
    PARTIAL = "partial"
    UNRELATED = "unrelated"


@dataclass
class FuncLocation:
    file: str
    name: str
    line_start: int
    line_end: int


@dataclass
class SimilarFunction:
    func_a: FuncLocation
    func_b: FuncLocation
    similarity_score: int  # 0-10
    similarity_type: SimilarityType
    analysis: str


@dataclass
class CompareResult:
    """Result of comparing two files."""
    file_a: str
    file_b: str
    similar_functions: list[SimilarFunction] = field(default_factory=list)


@dataclass
class ScdReport:
    repo_a_path: str
    repo_b_path: str
    repo_a_files: int
    repo_b_files: int
    dir_match_result: DirMatchResult | None = None
    compare_results: list[CompareResult] = field(default_factory=list)
    total_ai_calls: int = 0

    @property
    def all_similar_functions(self) -> list[SimilarFunction]:
        funcs: list[SimilarFunction] = []
        for cr in self.compare_results:
            funcs.extend(cr.similar_functions)
        return funcs
