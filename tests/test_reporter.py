from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scd.models import (
    CompareResult,
    DimensionScores,
    FuncLocation,
    ScdReport,
    SimilarFunction,
    SimilarityLevel,
)
from scd.reporter.markdown_template import extract_code_snippet, render_markdown
from scd.reporter.reporter import save_json


def _similar_function() -> SimilarFunction:
    return SimilarFunction(
        func_a=FuncLocation(file="src/a.py", name="normalize_user", line_start=2, line_end=4),
        func_b=FuncLocation(file="lib/b.py", name="normalize_user", line_start=3, line_end=5),
        composite_score=82,
        similarity_level=SimilarityLevel.HIGH,
        scores=DimensionScores(
            data_structure=80,
            function_signature=80,
            algorithm_logic=85,
            naming_convention=83,
            protocol_conformance=50,
        ),
        analysis="The implementations line up closely.",
    )


class ReporterCodeSnippetTests(unittest.TestCase):
    def test_extract_code_snippet_uses_file_and_line_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src" / "a.py"
            source.parent.mkdir()
            source.write_text(
                "before = 1\n"
                "def normalize_user(user):\n"
                "    name = user['name'].strip()\n"
                "    return {'name': name}\n"
                "after = 2\n",
                encoding="utf-8",
            )

            snippet = extract_code_snippet(str(root), "src/a.py", 2, 4)

            self.assertEqual(
                snippet,
                "2 | def normalize_user(user):\n"
                "3 |     name = user['name'].strip()\n"
                "4 |     return {'name': name}",
            )

    def test_markdown_report_includes_code_for_both_functions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            repo_a = Path(tmp_a)
            repo_b = Path(tmp_b)
            (repo_a / "src").mkdir()
            (repo_b / "lib").mkdir()
            (repo_a / "src" / "a.py").write_text(
                "x = 1\n"
                "def normalize_user(user):\n"
                "    name = user['name'].strip()\n"
                "    return {'name': name}\n",
                encoding="utf-8",
            )
            (repo_b / "lib" / "b.py").write_text(
                "x = 1\n"
                "y = 2\n"
                "def normalize_user(user):\n"
                "    name = user['name'].strip()\n"
                "    return {'name': name}\n",
                encoding="utf-8",
            )
            report = ScdReport(
                repo_a_path=str(repo_a),
                repo_b_path=str(repo_b),
                repo_a_files=1,
                repo_b_files=1,
                compare_results=[CompareResult("src/a.py", "lib/b.py", [_similar_function()])],
            )

            markdown = render_markdown(report)

            self.assertIn("**Code A:**", markdown)
            self.assertIn("2 | def normalize_user(user):", markdown)
            self.assertIn("4 |     return {'name': name}", markdown)
            self.assertIn("**Code B:**", markdown)
            self.assertIn("3 | def normalize_user(user):", markdown)
            self.assertIn("5 |     return {'name': name}", markdown)

    def test_json_report_includes_code_for_both_functions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            repo_a = Path(tmp_a)
            repo_b = Path(tmp_b)
            (repo_a / "src").mkdir()
            (repo_b / "lib").mkdir()
            (repo_a / "src" / "a.py").write_text(
                "x = 1\n"
                "def normalize_user(user):\n"
                "    name = user['name'].strip()\n"
                "    return {'name': name}\n",
                encoding="utf-8",
            )
            (repo_b / "lib" / "b.py").write_text(
                "x = 1\n"
                "y = 2\n"
                "def normalize_user(user):\n"
                "    name = user['name'].strip()\n"
                "    return {'name': name}\n",
                encoding="utf-8",
            )
            report = ScdReport(
                repo_a_path=str(repo_a),
                repo_b_path=str(repo_b),
                repo_a_files=1,
                repo_b_files=1,
                compare_results=[CompareResult("src/a.py", "lib/b.py", [_similar_function()])],
            )
            out = repo_a / "report.json"

            save_json(report, str(out))

            data = json.loads(out.read_text(encoding="utf-8"))
            pair = data["compare_results"][0]["similar_functions"][0]
            self.assertIn("2 | def normalize_user(user):", pair["func_a"]["code"])
            self.assertIn("3 | def normalize_user(user):", pair["func_b"]["code"])


if __name__ == "__main__":
    unittest.main()
