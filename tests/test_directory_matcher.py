from __future__ import annotations

import unittest

from scd.models import DirMatch
from scd.pipeline.directory_matcher import filter_matches_by_depth


def _match(dir_a: str, dir_b: str) -> DirMatch:
    return DirMatch(
        dir_a=dir_a,
        dir_b=dir_b,
        confidence="high",
        reason="test",
    )


class DirectoryMatchDepthTests(unittest.TestCase):
    def test_deepest_keeps_descendant_match_when_parent_and_child_overlap(self) -> None:
        matches = [
            _match("src", "lib"),
            _match("src/auth", "lib/security"),
            _match("docs", "manual"),
        ]

        filtered = filter_matches_by_depth(matches, "deepest")

        self.assertEqual(
            [(m.dir_a, m.dir_b) for m in filtered],
            [("src/auth", "lib/security"), ("docs", "manual")],
        )

    def test_highest_keeps_ancestor_match_when_parent_and_child_overlap(self) -> None:
        matches = [
            _match("src", "lib"),
            _match("src/auth", "lib/security"),
            _match("docs", "manual"),
        ]

        filtered = filter_matches_by_depth(matches, "highest")

        self.assertEqual(
            [(m.dir_a, m.dir_b) for m in filtered],
            [("src", "lib"), ("docs", "manual")],
        )

    def test_highest_treats_one_sided_ancestor_overlap_as_parent_claim(self) -> None:
        matches = [
            _match("drivers", "kernel/drivers"),
            _match("drivers/net", "networking"),
        ]

        filtered = filter_matches_by_depth(matches, "highest")

        self.assertEqual([(m.dir_a, m.dir_b) for m in filtered], [("drivers", "kernel/drivers")])


if __name__ == "__main__":
    unittest.main()
