from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scd.pipeline.dir_summarizer import SummaryCache
from scd.pipeline.file_summarizer import FileSummaryCache


class SummaryCachePathTests(unittest.TestCase):
    def test_file_summary_cache_uses_output_cache_dir(self) -> None:
        with tempfile.TemporaryDirectory() as repo_tmp, tempfile.TemporaryDirectory() as output_tmp:
            repo_path = Path(repo_tmp)
            output_cache_dir = Path(output_tmp) / ".scd_cache" / "repo_a"

            cache = FileSummaryCache(output_cache_dir, "model")

            self.assertEqual(cache.path, output_cache_dir / "file_summaries.jsonl")
            self.assertFalse((repo_path / ".scd_cache").exists())

    def test_directory_summary_cache_uses_output_cache_dir(self) -> None:
        with tempfile.TemporaryDirectory() as repo_tmp, tempfile.TemporaryDirectory() as output_tmp:
            repo_path = Path(repo_tmp)
            output_cache_dir = Path(output_tmp) / ".scd_cache" / "repo_b"

            cache = SummaryCache(output_cache_dir, "model")

            self.assertEqual(cache.path, output_cache_dir / "dir_summaries.jsonl")
            self.assertFalse((repo_path / ".scd_cache").exists())


if __name__ == "__main__":
    unittest.main()
