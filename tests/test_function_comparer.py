from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from scd.config import ScdConfig
from scd.models import DirInfo, DirMatch, FileInfo, RepoScanResult
from scd.pipeline.function_comparer import (
    PairCache,
    build_all_file_pairs,
    compare_file_pairs,
    pair_token_budget,
    split_code_into_chunks,
    _annotate_with_line_numbers,
    _estimate_code_tokens,
    _parse_similar_functions,
    _plan_pair_chunks,
)


def _repo(root: str, dir_path: str, files: list[tuple[str, str]]) -> RepoScanResult:
    file_infos = [
        FileInfo(path=path, language=language, line_count=10)
        for path, language in files
    ]
    return RepoScanResult(
        root_path=root,
        dirs={dir_path: DirInfo(path=dir_path, files=file_infos)},
        file_contents={path: "content" for path, _language in files},
    )


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def ask_json(self, _system: str, user: str, **_kwargs: object) -> dict:
        self.calls += 1
        if "src/b.c" in user:
            return {
                "similar_functions": [
                    {
                        "func_a": {"name": "copy_value", "line_start": 1, "line_end": 1},
                        "func_b": {"name": "copy_value", "line_start": 1, "line_end": 1},
                        "scores": {
                            "data_structure": 80,
                            "function_signature": 80,
                            "algorithm_logic": 80,
                            "naming_convention": 80,
                            "protocol_conformance": 80,
                        },
                        "analysis": "The implementation shape is aligned.",
                    }
                ]
            }
        return {"similar_functions": []}


class ParseSimilarFunctionsTests(unittest.TestCase):
    def test_filters_functionally_similar_but_not_visually_aligned_pair(self) -> None:
        data = {
            "similar_functions": [
                {
                    "func_a": {"name": "read_user", "line_start": 1, "line_end": 20},
                    "func_b": {"name": "fetch_account", "line_start": 4, "line_end": 31},
                    "scores": {
                        "data_structure": 25,
                        "function_signature": 40,
                        "algorithm_logic": 82,
                        "naming_convention": 20,
                        "protocol_conformance": 95,
                    },
                    "composite_score": 53,
                    "similarity_level": "medium",
                    "analysis": "Both fetch a user record over HTTP but use different payloads, names, and flow.",
                }
            ]
        }

        self.assertEqual(_parse_similar_functions(data, "a.py", "b.py", threshold=20), [])

    def test_keeps_pair_when_structure_logic_and_names_are_all_aligned(self) -> None:
        data = {
            "similar_functions": [
                {
                    "func_a": {"name": "normalize_user", "line_start": 3, "line_end": 18},
                    "func_b": {"name": "normalize_user", "line_start": 8, "line_end": 23},
                    "scores": {
                        "data_structure": 88,
                        "function_signature": 70,
                        "algorithm_logic": 90,
                        "naming_convention": 80,
                        "protocol_conformance": 50,
                    },
                    "composite_score": 88,
                    "similarity_level": "high",
                    "analysis": "Both functions use the same user dict fields, local names, and branch flow.",
                }
            ]
        }

        results = _parse_similar_functions(data, "a.py", "b.py", threshold=20)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].func_a.name, "normalize_user")

    def test_filters_pair_when_data_structure_is_below_gate(self) -> None:
        data = {
            "similar_functions": [
                {
                    "func_a": {"name": "normalize_user", "line_start": 3, "line_end": 18},
                    "func_b": {"name": "normalize_user", "line_start": 8, "line_end": 23},
                    "scores": {
                        "data_structure": 35,
                        "function_signature": 70,
                        "algorithm_logic": 74,
                        "naming_convention": 80,
                        "protocol_conformance": 50,
                    },
                    "composite_score": 66,
                    "similarity_level": "high",
                    "analysis": "The data shapes do not line up well enough.",
                }
            ]
        }

        self.assertEqual(_parse_similar_functions(data, "a.py", "b.py", threshold=20), [])

    def test_filters_pair_when_naming_convention_is_below_strict_gate(self) -> None:
        data = {
            "similar_functions": [
                {
                    "func_a": {"name": "normalize_user", "line_start": 3, "line_end": 18},
                    "func_b": {"name": "clean_record", "line_start": 8, "line_end": 23},
                    "scores": {
                        "data_structure": 88,
                        "function_signature": 50,
                        "algorithm_logic": 90,
                        "naming_convention": 20,
                        "protocol_conformance": 50,
                    },
                    "composite_score": 80,
                    "similarity_level": "high",
                    "analysis": "The data structures and implementation flow align despite different names.",
                }
            ]
        }

        self.assertEqual(_parse_similar_functions(data, "a.py", "b.py", threshold=20), [])

    def test_filters_pair_when_composite_is_below_strict_gate(self) -> None:
        data = {
            "similar_functions": [
                {
                    "func_a": {"name": "normalize_user", "line_start": 3, "line_end": 18},
                    "func_b": {"name": "normalize_user", "line_start": 8, "line_end": 23},
                    "scores": {
                        "data_structure": 85,
                        "function_signature": 50,
                        "algorithm_logic": 85,
                        "naming_convention": 50,
                        "protocol_conformance": 50,
                    },
                    "composite_score": 90,
                    "similarity_level": "high",
                    "analysis": "Core dimensions pass, but recomputed composite is below the strict gate.",
                }
            ]
        }

        self.assertEqual(_parse_similar_functions(data, "a.py", "b.py", threshold=20), [])

    def test_filters_pair_below_configured_threshold(self) -> None:
        data = {
            "similar_functions": [
                {
                    "func_a": {"name": "copy_user", "line_start": 1, "line_end": 10},
                    "func_b": {"name": "copy_user", "line_start": 1, "line_end": 10},
                    "scores": {
                        "data_structure": 60,
                        "function_signature": 50,
                        "algorithm_logic": 60,
                        "naming_convention": 40,
                        "protocol_conformance": 40,
                    },
                    "composite_score": 90,
                    "similarity_level": "high",
                    "analysis": "Recomputed composite (57) clears the core gate but is below the caller's threshold of 60.",
                }
            ]
        }

        self.assertEqual(_parse_similar_functions(data, "a.py", "b.py", threshold=60), [])

    def test_filters_pair_when_model_reports_b_function_from_a_file(self) -> None:
        data = {
            "similar_functions": [
                {
                    "func_a": {
                        "file": "a.py",
                        "name": "copy_user",
                        "line_start": 1,
                        "line_end": 10,
                    },
                    "func_b": {
                        "file": "a.py",
                        "name": "copy_user_fast",
                        "line_start": 12,
                        "line_end": 20,
                    },
                    "scores": {
                        "data_structure": 80,
                        "function_signature": 70,
                        "algorithm_logic": 80,
                        "naming_convention": 75,
                        "protocol_conformance": 60,
                    },
                    "analysis": "Both functions are from file A, so this is not a cross-file pair.",
                }
            ]
        }

        self.assertEqual(_parse_similar_functions(data, "a.py", "b.py", threshold=20), [])


def _fake_source(func_count: int, body_lines: int = 8) -> str:
    """Generate a Python-like file with column-0 function definitions."""
    parts: list[str] = []
    for i in range(func_count):
        parts.append(f"def func_{i}(value):")
        for j in range(body_lines):
            parts.append(f"    value = value + {j}  # step {j} of func_{i}")
        parts.append("    return value")
        parts.append("")
    return "\n".join(parts)


class ChunkingTests(unittest.TestCase):
    def test_annotate_keeps_original_line_numbers_for_chunks(self) -> None:
        annotated = _annotate_with_line_numbers("foo\nbar", start_line=42)
        lines = annotated.splitlines()
        self.assertTrue(lines[0].endswith("| foo"))
        self.assertTrue(lines[0].strip().startswith("42"))
        self.assertTrue(lines[1].strip().startswith("43"))

    def test_split_reconstructs_file_and_respects_budget(self) -> None:
        code = _fake_source(func_count=40)
        max_tokens = _estimate_code_tokens(code) // 4

        chunks = split_code_into_chunks(code, max_tokens)

        self.assertGreater(len(chunks), 1)
        # Chunks are contiguous, ordered, and cover the whole file.
        self.assertEqual(chunks[0].start_line, 1)
        for prev, cur in zip(chunks, chunks[1:]):
            self.assertEqual(cur.start_line, prev.end_line + 1)
        self.assertEqual(chunks[-1].end_line, len(code.splitlines()))
        # Trailing-newline-insensitive: chunking is line-based.
        self.assertEqual(
            "\n".join(c.code for c in chunks),
            "\n".join(code.splitlines()),
        )
        for chunk in chunks:
            self.assertLessEqual(_estimate_code_tokens(chunk.code), max_tokens)

    def test_split_prefers_function_boundaries(self) -> None:
        code = _fake_source(func_count=40)
        max_tokens = _estimate_code_tokens(code) // 4

        chunks = split_code_into_chunks(code, max_tokens)

        for chunk in chunks:
            first_nonempty = next(
                line for line in chunk.code.splitlines() if line.strip()
            )
            self.assertTrue(
                first_nonempty.startswith("def "),
                f"chunk starts mid-function: {first_nonempty!r}",
            )

    def test_small_pair_stays_whole(self) -> None:
        code_a = _fake_source(3)
        code_b = _fake_source(4)

        chunks_a, chunks_b = _plan_pair_chunks(
            code_a, code_b, pair_token_budget(128_000),
        )

        self.assertEqual(len(chunks_a), 1)
        self.assertEqual(len(chunks_b), 1)
        self.assertEqual(chunks_a[0].code, code_a)

    def test_only_oversized_side_is_split(self) -> None:
        code_a = _fake_source(60)
        code_b = _fake_source(2)
        budget = _estimate_code_tokens(code_a) // 2

        chunks_a, chunks_b = _plan_pair_chunks(code_a, code_b, budget)

        self.assertGreater(len(chunks_a), 1)
        self.assertEqual(len(chunks_b), 1)


class _ChunkAwareFakeClient:
    """Reports a similar pair only for the request containing both marker functions."""

    def __init__(self) -> None:
        self.calls = 0

    async def ask_json(self, _system: str, user: str, **_kwargs: object) -> dict:
        self.calls += 1
        if "def target_alpha" not in user or "def target_beta" not in user:
            return {"similar_functions": []}
        return {
            "similar_functions": [
                {
                    "func_a": {"name": "target_alpha", "line_start": 1, "line_end": 5},
                    "func_b": {"name": "target_beta", "line_start": 1, "line_end": 5},
                    "scores": {
                        "data_structure": 85,
                        "function_signature": 80,
                        "algorithm_logic": 85,
                        "naming_convention": 80,
                        "protocol_conformance": 80,
                    },
                    "analysis": "Both build the same dict and call the same helper.",
                }
            ]
        }


class ChunkedComparisonTests(unittest.TestCase):
    def test_oversized_pair_is_chunked_and_findings_are_merged(self) -> None:
        code_a = _fake_source(30) + "\ndef target_alpha(value):\n    return value\n"
        code_b = "def target_beta(value):\n    return value\n\n" + _fake_source(30)
        repo_a = RepoScanResult(
            root_path="repo_a",
            dirs={"src": DirInfo(path="src", files=[FileInfo("src/big.py", "python", 1)])},
            file_contents={"src/big.py": code_a},
        )
        repo_b = RepoScanResult(
            root_path="repo_b",
            dirs={"lib": DirInfo(path="lib", files=[FileInfo("lib/big.py", "python", 1)])},
            file_contents={"lib/big.py": code_b},
        )
        client = _ChunkAwareFakeClient()
        # Force chunking: budget only fits about half of each file per request.
        # pair_token_budget floors at 2000, so pick a window just above that.
        config = ScdConfig(context_window=11_500)

        results = asyncio.run(
            compare_file_pairs(
                [("src/big.py", "lib/big.py")], repo_a, repo_b, client, config,
            )
        )

        self.assertGreater(client.calls, 1)
        self.assertEqual(len(results), 1)
        names = {
            (sf.func_a.name, sf.func_b.name)
            for sf in results[0].similar_functions
        }
        self.assertEqual(names, {("target_alpha", "target_beta")})

    def test_chunk_results_are_cached_per_chunk(self) -> None:
        code_a = _fake_source(30)
        code_b = _fake_source(30)
        repo_a = RepoScanResult(
            root_path="repo_a",
            dirs={"src": DirInfo(path="src", files=[FileInfo("src/big.py", "python", 1)])},
            file_contents={"src/big.py": code_a},
        )
        repo_b = RepoScanResult(
            root_path="repo_b",
            dirs={"lib": DirInfo(path="lib", files=[FileInfo("lib/big.py", "python", 1)])},
            file_contents={"lib/big.py": code_b},
        )
        config = ScdConfig(context_window=11_500)
        pair = [("src/big.py", "lib/big.py")]

        with tempfile.TemporaryDirectory() as tmp:
            first_client = _ChunkAwareFakeClient()
            asyncio.run(
                compare_file_pairs(
                    pair, repo_a, repo_b, first_client, config,
                    cache=PairCache(tmp),
                )
            )
            self.assertGreater(first_client.calls, 1)

            second_cache = PairCache(tmp)
            second_cache.load()
            second_client = _ChunkAwareFakeClient()
            asyncio.run(
                compare_file_pairs(
                    pair, repo_a, repo_b, second_client, config,
                    cache=second_cache,
                )
            )
            self.assertEqual(second_client.calls, 0)


class BuildFilePairsTests(unittest.TestCase):
    def test_keeps_only_same_extension_pairs_for_matched_directories(self) -> None:
        repo_a = _repo(
            "repo_a",
            "src",
            [
                ("src/main.c", "c"),
                ("src/api.h", "c"),
            ],
        )
        repo_b = _repo(
            "repo_b",
            "lib",
            [
                ("lib/main.c", "c"),
                ("lib/api.h", "c"),
            ],
        )
        match = DirMatch("src", "lib", confidence="high", reason="test")

        pairs = build_all_file_pairs([match], repo_a, repo_b)

        self.assertEqual(
            pairs,
            [
                ("src/main.c", "lib/main.c"),
                ("src/api.h", "lib/api.h"),
            ],
        )


class CompareFilePairsTests(unittest.TestCase):
    def test_calls_on_result_only_when_similar_functions_are_found(self) -> None:
        repo_a = _repo(
            "repo_a",
            "src",
            [
                ("src/a.c", "c"),
                ("src/b.c", "c"),
            ],
        )
        repo_b = _repo(
            "repo_b",
            "lib",
            [
                ("lib/a.c", "c"),
                ("lib/b.c", "c"),
            ],
        )
        file_pairs = [
            ("src/a.c", "lib/a.c"),
            ("src/b.c", "lib/b.c"),
        ]
        seen: list[tuple[str, str]] = []

        results = asyncio.run(
            compare_file_pairs(
                file_pairs,
                repo_a,
                repo_b,
                _FakeClient(),
                ScdConfig(),
                on_similar_result=lambda result: seen.append((result.file_a, result.file_b)),
            )
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(seen, [("src/b.c", "lib/b.c")])

    def test_thresholds_below_strict_floor_share_pair_cache_key(self) -> None:
        repo_a = _repo("repo_a", "src", [("src/b.c", "c")])
        repo_b = _repo("repo_b", "lib", [("lib/b.c", "c")])
        file_pairs = [("src/b.c", "lib/b.c")]

        with tempfile.TemporaryDirectory() as tmp:
            first_cache = PairCache(tmp)
            first_client = _FakeClient()
            asyncio.run(
                compare_file_pairs(
                    file_pairs,
                    repo_a,
                    repo_b,
                    first_client,
                    ScdConfig(similarity_threshold=20),
                    cache=first_cache,
                )
            )

            second_cache = PairCache(tmp)
            second_cache.load()
            second_client = _FakeClient()
            asyncio.run(
                compare_file_pairs(
                    file_pairs,
                    repo_a,
                    repo_b,
                    second_client,
                    ScdConfig(similarity_threshold=80),
                    cache=second_cache,
                )
            )

        self.assertEqual(first_client.calls, 1)
        self.assertEqual(second_client.calls, 0)

    def test_pair_cache_load_skips_records_from_old_cache_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = PairCache(tmp)
            cache.path.write_text(
                json.dumps(
                    {
                        "v": 3,
                        "key": "legacy",
                        "file_a": "src/a.c",
                        "file_b": "lib/a.c",
                        "similar_functions": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(cache.load(), 0)
            self.assertIsNone(cache.get("legacy"))


if __name__ == "__main__":
    unittest.main()
