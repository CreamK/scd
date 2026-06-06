from __future__ import annotations

import unittest

from scd.models import DirInfo, DirMatch, FileInfo, RepoScanResult
from scd.pipeline.function_comparer import build_all_file_pairs, _parse_similar_functions


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
                        "data_structure": 76,
                        "function_signature": 70,
                        "algorithm_logic": 74,
                        "naming_convention": 80,
                        "protocol_conformance": 50,
                    },
                    "composite_score": 72,
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

    def test_keeps_pair_when_only_naming_convention_is_below_gate(self) -> None:
        data = {
            "similar_functions": [
                {
                    "func_a": {"name": "normalize_user", "line_start": 3, "line_end": 18},
                    "func_b": {"name": "clean_record", "line_start": 8, "line_end": 23},
                    "scores": {
                        "data_structure": 70,
                        "function_signature": 50,
                        "algorithm_logic": 74,
                        "naming_convention": 20,
                        "protocol_conformance": 50,
                    },
                    "composite_score": 64,
                    "similarity_level": "high",
                    "analysis": "The data structures and implementation flow align despite different names.",
                }
            ]
        }

        results = _parse_similar_functions(data, "a.py", "b.py", threshold=20)

        self.assertEqual(len(results), 1)

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


if __name__ == "__main__":
    unittest.main()
