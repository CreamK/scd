from __future__ import annotations

import json
import hashlib
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
from scd.reporter.reporter import save_json, save_reports


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

    def test_json_report_uses_review_array_with_html_context_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            repo_a = Path(tmp_a)
            repo_b = Path(tmp_b)
            (repo_a / "src").mkdir()
            (repo_b / "lib").mkdir()
            content_a = (
                "far_before = 0\n"
                + "".join(f"before_a_{i} = {i}\n" for i in range(1, 23))
                + "def normalize_user(user):\n"
                "    name = user['name'].strip()\n"
                "    return {'name': name, 'tag': '<admin>'}\n"
                + "".join(f"after_a_{i} = {i}\n" for i in range(1, 23))
                + "far_after = 99\n"
            )
            content_b = (
                "far_before = 0\n"
                + "".join(f"before_b_{i} = {i}\n" for i in range(1, 24))
                + "def normalize_user(user):\n"
                "    name = user['name'].strip()\n"
                "    return {'name': name, 'tag': '<admin>'}\n"
                + "".join(f"after_b_{i} = {i}\n" for i in range(1, 23))
                + "far_after = 99\n"
            )
            (repo_a / "src" / "a.py").write_text(content_a, encoding="utf-8")
            (repo_b / "lib" / "b.py").write_text(content_b, encoding="utf-8")
            report = ScdReport(
                repo_a_path=str(repo_a),
                repo_b_path=str(repo_b),
                repo_a_files=1,
                repo_b_files=1,
                compare_results=[
                    CompareResult(
                        "src/a.py",
                        "lib/b.py",
                        [
                            SimilarFunction(
                                func_a=FuncLocation(
                                    file="src/a.py",
                                    name="normalize_user",
                                    line_start=24,
                                    line_end=26,
                                ),
                                func_b=FuncLocation(
                                    file="lib/b.py",
                                    name="normalize_user",
                                    line_start=25,
                                    line_end=27,
                                ),
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
                        ],
                    )
                ],
            )
            out = repo_a / "report.json"

            save_json(report, str(out))

            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)
            pair = data[0]
            self.assertEqual(
                set(pair),
                {"huawei_file", "Linux_file", "html_context", "reason", "severity", "hash"},
            )
            self.assertEqual(pair["huawei_file"], "src/a.py")
            self.assertEqual(pair["Linux_file"], "lib/b.py")
            self.assertEqual(pair["reason"], "The implementations line up closely.")
            self.assertEqual(pair["severity"], "high")
            func_source_a = "\n".join(content_a.splitlines()[23:26])
            func_source_b = "\n".join(content_b.splitlines()[24:27])
            expected_hasher = hashlib.sha256()
            expected_hasher.update(func_source_a.encode("utf-8"))
            expected_hasher.update(b"\x00")
            expected_hasher.update(func_source_b.encode("utf-8"))
            self.assertEqual(pair["hash"], expected_hasher.hexdigest())
            self.assertIn('class="scd-side scd-side-left"', pair["html_context"])
            self.assertIn('class="scd-highlight"', pair["html_context"])
            self.assertIn("background-color:#ffe4e6;", pair["html_context"])
            self.assertIn("background-color:#dcfce7;", pair["html_context"])
            self.assertIn("def normalize_user(user):", pair["html_context"])
            self.assertIn("&lt;admin&gt;", pair["html_context"])
            self.assertIn("before_a_3 = 3", pair["html_context"])
            self.assertNotIn("far_before = 0", pair["html_context"])
            self.assertNotIn("far_after = 99", pair["html_context"])

    def test_save_reports_writes_markdown_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            repo_a = Path(tmp_a)
            repo_b = Path(tmp_b)
            (repo_a / "src").mkdir()
            (repo_b / "lib").mkdir()
            (repo_a / "src" / "a.py").write_text(
                "x = 1\n"
                "def normalize_user(user):\n"
                "    return user\n",
                encoding="utf-8",
            )
            (repo_b / "lib" / "b.py").write_text(
                "x = 1\n"
                "y = 2\n"
                "def normalize_user(user):\n"
                "    return user\n",
                encoding="utf-8",
            )
            report = ScdReport(
                repo_a_path=str(repo_a),
                repo_b_path=str(repo_b),
                repo_a_files=1,
                repo_b_files=1,
                compare_results=[CompareResult("src/a.py", "lib/b.py", [_similar_function()])],
            )
            md_path = repo_a / "report.md"
            json_path = repo_a / "report.json"

            save_reports(report, str(md_path), str(json_path))

            self.assertIn("# SCD - Code Similarity Report", md_path.read_text(encoding="utf-8"))
            self.assertIsInstance(json.loads(json_path.read_text(encoding="utf-8")), list)


if __name__ == "__main__":
    unittest.main()
