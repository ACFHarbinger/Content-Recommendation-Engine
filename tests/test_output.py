"""
Tests for src/output.py — print_table and to_json renderers.

Uses scored_candidates from conftest.py fed through the real Scorer.
"""

from __future__ import annotations

import json

from src.core.schema import ComponentScores, ExplainedResult, MediaItem # pyrefly: ignore [missing-import]


def _make_explained(
    item: MediaItem, rank: int, rv: float, reasons=None, matched_tags=None
) -> ExplainedResult:
    return ExplainedResult(
        item=item,
        recommendation_value=rv,
        component_scores=ComponentScores(
            rrf_score=rv,
            rating_boost=1.2,
            recency_decay=0.9,
            length_decay=1.0,
        ),
        rank=rank,
        reasons=reasons or ["Matches the query well."],
        matched_tags=matched_tags or [],
    )


class TestToJson:
    def test_returns_valid_json_string(self, cfg, scored_candidates):
        from src.cli.output import to_json # pyrefly: ignore [missing-import]
        from src.search.scorer import Scorer # pyrefly: ignore [missing-import]

        ranked = Scorer(cfg).score(scored_candidates)
        explained = [
            ExplainedResult(**r.model_dump(), reasons=["test"], matched_tags=[])
            for r in ranked
        ]
        raw = to_json(explained)
        data = json.loads(raw)
        assert isinstance(data, list)
        assert len(data) == len(explained)

    def test_json_contains_required_keys(self, cfg, scored_candidates):
        from src.cli.output import to_json # pyrefly: ignore [missing-import]
        from src.search.scorer import Scorer # pyrefly: ignore [missing-import]

        ranked = Scorer(cfg).score(scored_candidates[:1])
        explained = [
            ExplainedResult(**r.model_dump(), reasons=["r"], matched_tags=[])
            for r in ranked
        ]
        data = json.loads(to_json(explained))
        row = data[0]
        for key in (
            "rank",
            "title",
            "type",
            "year",
            "rating",
            "recommendation_value",
            "component_scores",
            "reasons",
            "matched_tags",
            "genres",
            "tags",
        ):
            assert key in row, f"Missing key: {key}"

    def test_json_component_scores_nested(self, cfg, scored_candidates):
        from src.cli.output import to_json # pyrefly: ignore [missing-import]
        from src.search.scorer import Scorer # pyrefly: ignore [missing-import]

        ranked = Scorer(cfg).score(scored_candidates[:1])
        explained = [
            ExplainedResult(**r.model_dump(), reasons=[], matched_tags=[])
            for r in ranked
        ]
        data = json.loads(to_json(explained))
        cs = data[0]["component_scores"]
        assert "rrf_score" in cs
        assert "rating_boost" in cs
        assert "recency_decay" in cs
        assert "length_decay" in cs

    def test_empty_results_returns_empty_json_array(self):
        from src.cli.output import to_json # pyrefly: ignore [missing-import]

        assert to_json([]) == "[]"

    def test_indent_parameter(self, cfg, scored_candidates):
        from src.cli.output import to_json # pyrefly: ignore [missing-import]
        from src.search.scorer import Scorer # pyrefly: ignore [missing-import]

        ranked = Scorer(cfg).score(scored_candidates[:1])
        explained = [
            ExplainedResult(**r.model_dump(), reasons=[], matched_tags=[])
            for r in ranked
        ]
        compact = to_json(explained, indent=None)
        pretty = to_json(explained, indent=4)
        assert len(pretty) > len(compact)


class TestPrintTable:
    """Tests that print_table runs without errors and produces output."""

    def test_prints_without_error(self, capsys, cfg, scored_candidates):
        from src.cli.output import print_table # pyrefly: ignore [missing-import]
        from src.search.scorer import Scorer # pyrefly: ignore [missing-import]

        ranked = Scorer(cfg).score(scored_candidates)
        explained = [
            ExplainedResult(
                **r.model_dump(), reasons=["match"], matched_tags=["cyberpunk"]
            )
            for r in ranked
        ]
        print_table(explained)  # should not raise

    def test_empty_results_does_not_crash(self, capsys):
        from src.cli.output import print_table # pyrefly: ignore [missing-import]

        print_table([])  # should not raise

    def test_handles_missing_links(self, cfg, sample_items):
        from src.cli.output import print_table # pyrefly: ignore [missing-import]

        item = MediaItem.model_validate(
            {
                "id": "x",
                "title": "No Links",
                "type": "anime",
                "rating": 7.0,
                "year": 2010,
                "episodes": 12,
            }
        )
        explained = [_make_explained(item, rank=1, rv=0.7)]
        print_table(explained)  # should not raise on missing web_link/local_file

    def test_score_bar_bounds(self):
        from src.cli.output import _score_bar # pyrefly: ignore [missing-import]

        bar = _score_bar(0.0)
        assert "█" not in bar
        bar = _score_bar(1.0)
        assert "░" not in bar

    def test_type_badge_unknown_type(self):
        from src.cli.output import _type_badge # pyrefly: ignore [missing-import]

        badge = _type_badge("unknown_type")
        assert badge.plain == "UNKNOWN_TYPE"

    def test_type_badge_known_types(self):
        from src.cli.output import _type_badge # pyrefly: ignore [missing-import]

        for t in ("anime", "movie", "book", "manga", "show", "paper"):
            badge = _type_badge(t)
            assert badge.plain == t.upper()
