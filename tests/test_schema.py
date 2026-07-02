"""
Phase 0 acceptance test: validate data/sample.json against MediaItem schema.

Run: pytest tests/test_schema.py
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.schema import ExplainedResult, MediaItem, ParsedQuery

SAMPLE_PATH = Path(__file__).parent.parent / "data" / "sample.json"


def load_sample() -> list[dict]:
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        return json.load(f)


class TestMediaItemValidation:
    def test_sample_file_exists(self):
        assert SAMPLE_PATH.exists(), f"sample.json not found at {SAMPLE_PATH}"

    def test_all_sample_items_are_valid(self):
        raw = load_sample()
        errors = []
        for i, item in enumerate(raw):
            try:
                MediaItem.model_validate(item)
            except ValidationError as e:
                errors.append(f"Item {i} ({item.get('id')}): {e}")
        assert not errors, "\n".join(errors)

    def test_item_count(self):
        raw = load_sample()
        assert len(raw) >= 5, "Sample file should have at least 5 items"

    def test_csv_genres_coerced_to_list(self):
        item = MediaItem.model_validate(
            {"id": "abc", "title": "Test", "genres": "Action, Drama, Sci-Fi"}
        )
        assert item.genres == ["Action", "Drama", "Sci-Fi"]

    def test_list_genres_accepted(self):
        item = MediaItem.model_validate(
            {"id": "abc", "title": "Test", "genres": ["Action", "Drama"]}
        )
        assert item.genres == ["Action", "Drama"]

    def test_rating_bounds(self):
        with pytest.raises(ValidationError):
            MediaItem.model_validate({"id": "x", "title": "Bad", "rating": 11.0})
        with pytest.raises(ValidationError):
            MediaItem.model_validate({"id": "x", "title": "Bad", "rating": -1.0})

    def test_dense_text_falls_back_to_concat(self):
        item = MediaItem.model_validate(
            {
                "id": "x",
                "title": "Evangelion",
                "genres": "Mecha",
                "tags": "depression",
            }
        )
        assert item.dense_text  # Should never be empty
        assert "Evangelion" in item.dense_text

    def test_dense_text_uses_review_when_present(self):
        item = MediaItem.model_validate(
            {"id": "x", "title": "Test", "review": "Great anime about robots."}
        )
        assert item.dense_text == "Great anime about robots."

    def test_sparse_text_includes_title_genres_tags(self):
        item = MediaItem.model_validate(
            {
                "id": "x",
                "title": "Ghost in the Shell",
                "genres": ["Cyberpunk", "Sci-Fi"],
                "tags": ["AI", "consciousness"],
            }
        )
        sparse = item.sparse_text
        assert "Ghost in the Shell" in sparse
        assert "Cyberpunk" in sparse
        assert "AI" in sparse

    def test_none_rating_allowed(self):
        item = MediaItem.model_validate({"id": "y", "title": "Unrated", "rating": None})
        assert item.rating is None


class TestExplainedResultValidation:
    def test_matched_tags_stripped_when_not_in_item(self):
        item = MediaItem.model_validate(
            {"id": "z", "title": "T", "tags": ["mecha"], "genres": ["Action"]}
        )
        from src.core.schema import ComponentScores, RankedResult

        ranked = RankedResult(
            item=item,
            recommendation_value=0.8,
            component_scores=ComponentScores(
                rrf_score=0.5,
                rating_boost=1.0,
                recency_decay=1.0,
                length_decay=1.0,
            ),
            rank=1,
        )
        explained = ExplainedResult(
            **ranked.model_dump(),
            reasons=["Because mecha"],
            matched_tags=["mecha", "fake-tag"],  # "fake-tag" should be stripped
        )
        assert "fake-tag" not in explained.matched_tags
        assert "mecha" in explained.matched_tags


class TestParsedQuery:
    def test_empty_query_defaults(self):
        q = ParsedQuery()
        assert q.semantic_query == ""
        assert q.filters == []
        assert q.length_preference_episodes is None
