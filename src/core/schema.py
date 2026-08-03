"""
Pydantic v2 data models for the Recommendation Engine.

Phase 0: foundation schema.
Phase 1: EmbeddedItem, ScoredCandidate added for ingestion pipeline.
Phase 2: FilterClause, ParsedQuery.
Phase 3: ScoredCandidate with per-leg scores.
Phase 4: RankedResult with recommendation_value.
Phase 5: ExplainedResult with reasons and anti-hallucination guard.
Phase 7: volumes, abstract fields on MediaItem; modality-aware dense_text.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ------------------------------------------------------------------
# Enumerations
# ------------------------------------------------------------------


class MediaType(str, Enum):
    ANIME = "anime"
    SHOW = "show"
    MOVIE = "movie"
    BOOK = "book"
    MANGA = "manga"
    PAPER = "paper"
    GAME = "game"
    OTHER = "other"


class WatchStatus(str, Enum):
    WATCHED = "watched"
    READING = "reading"
    PLAN_TO_WATCH = "plan_to_watch"
    ON_HOLD = "on_hold"
    DROPPED = "dropped"


# ------------------------------------------------------------------
# Core media item
# ------------------------------------------------------------------


class MediaItem(BaseModel):
    """Single item in the media library.  Matches the listings.json schema."""

    id: str = Field(..., description="UUID v4 stable identifier")
    title: str = Field(..., min_length=1)
    type: Optional[str] = Field(None, description="anime|show|movie|book|manga|paper|…")
    watch_status: Optional[str] = Field(None, alias="status")
    rating: Optional[float] = Field(None, ge=0.0, le=10.0)
    year_released: Optional[int] = Field(None, alias="year", ge=1800, le=2200)
    num_episodes_or_pages: Optional[int] = Field(None, alias="episodes", ge=0)
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    associated_entities: list[str] = Field(default_factory=list)
    local_file_location: Optional[str] = Field(None, alias="local_file")
    web_link: Optional[str] = None
    review_notes: Optional[str] = Field(None, alias="review")
    # Phase 7 additions
    abstract: Optional[str] = None  # for papers; used as dense source
    volumes: Optional[int] = Field(None, ge=1)  # for manga

    model_config = {"populate_by_name": True, "extra": "allow"}

    @field_validator("genres", "tags", "associated_entities", mode="before")
    @classmethod
    def coerce_csv_to_list(cls, v: Any) -> list[str]:
        """Accept either a list or a comma-separated string."""
        if isinstance(v, list):
            return [s.strip() for s in v if str(s).strip()]
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return []

    @property
    def dense_text(self) -> str:
        """Primary text for dense (semantic) embedding.
        Priority: review_notes → abstract → synthesised fallback."""
        return self.review_notes or self.abstract or self._fallback_text()

    @property
    def sparse_text(self) -> str:
        """Text for sparse (lexical) embedding."""
        parts = [self.title]
        if self.genres:
            parts.append("Genres: " + ", ".join(self.genres))
        if self.tags:
            parts.append("Tags: " + ", ".join(self.tags))
        if self.associated_entities:
            parts.append("Featuring: " + ", ".join(self.associated_entities))
        return " ".join(parts)

    @property
    def is_video(self) -> bool:
        return (self.type or "").lower() in ("anime", "show", "movie")

    @property
    def consume_verb(self) -> str:
        """Modality-appropriate consumption verb for explainer prose."""
        t = (self.type or "").lower()
        if t in ("book", "manga", "paper"):
            return "read"
        return "watch"

    def _fallback_text(self) -> str:
        parts = [self.title]
        if self.type:
            parts.append(f"Type: {self.type}")
        parts.extend(self.genres)
        parts.extend(self.tags)
        parts.extend(self.associated_entities)
        return " ".join(parts)


# ------------------------------------------------------------------
# Watch-history profile (Phase 9)
# ------------------------------------------------------------------


class HistoryProfile(BaseModel):
    """
    Aggregated preference signal derived from the user's watch history.

    Built by scanning the Qdrant collection for items with
    watch_status == "watched" (or "reading") and rating >= threshold,
    then collecting their tags and genres.

    The profile is passed to Scorer to apply a mild boost to candidates
    that share tags/genres with the user's proven preferences.
    """

    preferred_tags: frozenset[str] = Field(default_factory=frozenset)
    preferred_genres: frozenset[str] = Field(default_factory=frozenset)
    item_count: int = 0  # number of items that contributed

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_payloads(cls, payloads: list[dict]) -> "HistoryProfile":
        tags: set[str] = set()
        genres: set[str] = set()
        for p in payloads:
            tags.update(t.lower() for t in p.get("tags", []))
            genres.update(g.lower() for g in p.get("genres", []))
        return cls(
            preferred_tags=frozenset(tags),
            preferred_genres=frozenset(genres),
            item_count=len(payloads),
        )

    def overlap_fraction(self, item: "MediaItem") -> float:
        """Fraction of item's tags+genres that appear in this profile (0–1)."""
        item_signals = {t.lower() for t in item.tags + item.genres}
        if not item_signals:
            return 0.0
        profile = self.preferred_tags | self.preferred_genres
        return len(item_signals & profile) / len(item_signals)


# ------------------------------------------------------------------
# Ingestion outputs (Phase 1)
# ------------------------------------------------------------------


class EmbeddedItem(BaseModel):
    item: MediaItem
    dense_vector: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


# ------------------------------------------------------------------
# Query / retrieval (Phase 2–3)
# ------------------------------------------------------------------


class FilterClause(BaseModel):
    field: str
    op: str  # eq, ne, gt, gte, lt, lte, in, nin
    value: Any


class ParsedQuery(BaseModel):
    semantic_query: str = ""
    filters: list[FilterClause] = Field(default_factory=list)
    length_preference_episodes: Optional[int] = None
    raw_filters: list[dict] = Field(default_factory=list)


# ------------------------------------------------------------------
# Scored / ranked results (Phases 3–4)
# ------------------------------------------------------------------


class ScoredCandidate(BaseModel):
    item: MediaItem
    rrf_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0


class ComponentScores(BaseModel):
    rrf_score: float
    rating_boost: float
    recency_decay: float
    length_decay: float


class RankedResult(BaseModel):
    item: MediaItem
    recommendation_value: float
    component_scores: ComponentScores
    rank: int


# ------------------------------------------------------------------
# Explained results (Phase 5)
# ------------------------------------------------------------------


class ExplainedResult(RankedResult):
    reasons: list[str] = Field(default_factory=list)
    matched_tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_matched_tags(self) -> "ExplainedResult":
        """Strip any tags that aren't in the item's actual tag/genre lists."""
        valid = set(self.item.tags) | set(self.item.genres)
        self.matched_tags = [t for t in self.matched_tags if t in valid]
        return self
