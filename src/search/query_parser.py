"""
Phase 2 — Self-Querying Retriever: converts free-form user prompts into
(semantic_query, SQL filter) using the Claude API.

Falls back to a pure semantic search (no filters, full query as semantic text)
on any API error, malformed JSON, or missing API key — so the pipeline never
crashes due to a parsing failure.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from src.core.cache import QueryCache  # pyrefly: ignore [missing-import]
from src.core.config import Settings, get_settings  # pyrefly: ignore [missing-import]
from src.core.schema import FilterClause, ParsedQuery  # pyrefly: ignore [missing-import]

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Few-shot examples embedded directly in the system prompt
# -----------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a structured query parser for a personal media recommendation engine.
The media library contains items with these filterable fields:

  type            : string  — anime | show | movie | book | manga | paper | game | other
  watch_status    : string  — watched | reading | plan_to_watch | on_hold | dropped
  rating          : float   — 0.0 – 10.0 (personal rating)
  year_released   : int     — calendar year (e.g. 2019)
  num_episodes_or_pages : int — episode count for video, page count for books
  genres          : string[] — e.g. ["Action", "Sci-Fi"]
  tags            : string[] — e.g. ["time-travel", "mecha"]

Allowed operators:
  eq  — exact match (use for type, watch_status, single-value fields)
  ne  — not equal
  gt / gte / lt / lte — numeric comparisons
  in  — value is in a list  (use for multi-value fields like genres/tags)
  nin — value is NOT in a list

Parse the user query and return ONLY valid JSON matching this schema:
{
  "semantic_query": "<the descriptive/thematic part, empty string if purely structural>",
  "filters": [
    {"field": "<field>", "op": "<op>", "value": <value>}
  ],
  "length_preference_episodes": <int or null>
}

Rules:
1. Extract ALL structural constraints as filters; keep only the thematic description in semantic_query.
2. If the query is purely structural with no semantic content, semantic_query = "".
3. Never invent fields or operators not listed above.
4. length_preference_episodes: set only when the user explicitly expresses an episode/length preference.
5. Output ONLY the JSON object — no markdown, no explanation.

--- EXAMPLES ---

User: "something like Evangelion but less depressing"
{"semantic_query":"something like Neon Genesis Evangelion but with a less depressing tone","filters":[],"length_preference_episodes":null}

User: "all anime I haven't watched yet"
{"semantic_query":"","filters":[{"field":"type","op":"eq","value":"anime"},{"field":"watch_status","op":"nin","value":["watched","dropped"]}],"length_preference_episodes":null}

User: "short highly-rated mecha from the 90s I haven't seen"
{"semantic_query":"mecha giant robots action","filters":[{"field":"watch_status","op":"ne","value":"watched"},{"field":"rating","op":"gte","value":8.0},{"field":"year_released","op":"gte","value":1990},{"field":"year_released","op":"lte","value":1999},{"field":"tags","op":"in","value":["mecha"]}],"length_preference_episodes":26}

User: "psychological thriller anime rated above 8 I haven't seen"
{"semantic_query":"psychological thriller dark atmosphere","filters":[{"field":"type","op":"eq","value":"anime"},{"field":"watch_status","op":"nin","value":["watched","dropped"]},{"field":"rating","op":"gte","value":8.0},{"field":"genres","op":"in","value":["Psychological","Thriller"]}],"length_preference_episodes":null}

User: "manga with over 30 volumes"
{"semantic_query":"","filters":[{"field":"type","op":"eq","value":"manga"},{"field":"num_episodes_or_pages","op":"gt","value":30}],"length_preference_episodes":null}

User: "CS paper about attention mechanisms in transformers"
{"semantic_query":"attention mechanisms transformer architecture self-attention","filters":[{"field":"type","op":"eq","value":"paper"}],"length_preference_episodes":null}

User: "classic sci-fi movies from before 2000"
{"semantic_query":"classic science fiction space exploration futurism","filters":[{"field":"type","op":"eq","value":"movie"},{"field":"year_released","op":"lt","value":2000}],"length_preference_episodes":null}

User: "show me everything on my plan-to-watch list"
{"semantic_query":"","filters":[{"field":"watch_status","op":"eq","value":"plan_to_watch"}],"length_preference_episodes":null}

User: "dark fantasy books I'm currently reading"
{"semantic_query":"dark fantasy magic world-building","filters":[{"field":"type","op":"eq","value":"book"},{"field":"watch_status","op":"eq","value":"reading"}],"length_preference_episodes":null}

User: "space opera anime under 26 episodes with action"
{"semantic_query":"space opera interstellar adventure action","filters":[{"field":"type","op":"eq","value":"anime"},{"field":"genres","op":"in","value":["Action","Sci-Fi"]}],"length_preference_episodes":26}
"""

# Fields whose values are stored as JSON arrays in SQLite
_ARRAY_FIELDS = frozenset({"genres", "tags", "associated_entities"})


def _build_array_clause(field: str, op: str, value: Any) -> tuple[str, list]:
    vals = value if isinstance(value, list) else [value]
    if op == "in":
        ph = ",".join("?" * len(vals))
        return f"EXISTS (SELECT 1 FROM json_each({field}) WHERE value IN ({ph}))", vals
    if op == "nin":
        ph = ",".join("?" * len(vals))
        return f"NOT EXISTS (SELECT 1 FROM json_each({field}) WHERE value IN ({ph}))", vals
    if op == "eq":
        return f"EXISTS (SELECT 1 FROM json_each({field}) WHERE value = ?)", [value]
    if op == "ne":
        return f"NOT EXISTS (SELECT 1 FROM json_each({field}) WHERE value = ?)", [value]
    return "", []


def _build_scalar_clause(field: str, op: str, value: Any) -> tuple[str, list]:
    if op == "eq":
        return f"{field} = ?", [value]
    if op == "ne":
        return f"{field} != ?", [value]
    if op == "gt":
        return f"{field} > ?", [value]
    if op == "gte":
        return f"{field} >= ?", [value]
    if op == "lt":
        return f"{field} < ?", [value]
    if op == "lte":
        return f"{field} <= ?", [value]
    if op in ("in", "nin"):
        vals = value if isinstance(value, list) else [value]
        ph = ",".join("?" * len(vals))
        if op == "in":
            return f"{field} IN ({ph})", vals
        return f"{field} NOT IN ({ph})", vals
    return "", []


def _build_sql_filter(filters: list[FilterClause]) -> tuple[str, list]:
    """
    Convert ParsedQuery filters into a SQL WHERE clause.

    Returns (where_clause, params) where where_clause is a SQL condition
    string without the WHERE keyword, or ("", []) when filters is empty.
    """
    if not filters:
        return "", []

    clauses: list[str] = []
    params: list[Any] = []

    for f in filters:
        field, op, value = f.field, f.op, f.value

        if field in _ARRAY_FIELDS:
            clause, vals = _build_array_clause(field, op, value)
        else:
            clause, vals = _build_scalar_clause(field, op, value)

        if clause:
            clauses.append(clause)
            params.extend(vals)

    return " AND ".join(clauses), params


class QueryParser:
    """
    Uses the Claude API (claude-sonnet-4-6 by default) to parse a free-form
    user query into structured retrieval parameters.

    Falls back to pure semantic search on any failure.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._cfg = settings or get_settings()
        self._cache = QueryCache()

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse *query* → ParsedQuery.  Checks cache first; calls Claude on miss.
        Falls back to ``ParsedQuery(semantic_query=query)`` on any error.
        """
        cached = self._cache.get(query)
        if cached is not None:
            logger.debug("Cache hit for query: %r", query[:60])
            return cached

        parsed = self._call_claude(query)
        self._cache.put(query, parsed)
        return parsed

    def parse_to_sql(self, query: str) -> tuple[ParsedQuery, tuple[str, list]]:
        """Convenience: parse query and return (ParsedQuery, (where_clause, params))."""
        parsed = self.parse(query)
        sql_filter = _build_sql_filter(parsed.filters)
        return parsed, sql_filter

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_claude(self, query: str) -> ParsedQuery:
        if not self._cfg.anthropic_api_key:
            logger.info("No ANTHROPIC_API_KEY set — falling back to semantic-only.")
            return ParsedQuery(semantic_query=query)

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self._cfg.anthropic_api_key)
            # Cache the static system prompt so repeated calls hit the cache.
            system_block = [
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            # pyrefly: ignore [no-matching-overload]
            response = client.messages.create(
                model=self._cfg.claude_model,
                max_tokens=512,
                system=system_block,
                messages=[{"role": "user", "content": query}],
            )
            raw_text = response.content[0].text.strip()
            return self._parse_response(raw_text, query)

        except Exception as exc:
            logger.warning("Claude API call failed (%s) — falling back.", exc)
            return ParsedQuery(semantic_query=query)

    def _parse_response(self, text: str, original_query: str) -> ParsedQuery:
        """Extract JSON from Claude's response and build a ParsedQuery."""
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Malformed JSON from Claude (%s) — falling back.", exc)
            return ParsedQuery(semantic_query=original_query)

        try:
            raw_filters = data.get("filters", [])
            filter_clauses = [FilterClause(**f) for f in raw_filters]
            return ParsedQuery(
                semantic_query=data.get("semantic_query", ""),
                filters=filter_clauses,
                length_preference_episodes=data.get("length_preference_episodes"),
                raw_filters=raw_filters,
            )
        except Exception as exc:
            logger.warning("Invalid filter structure (%s) — falling back.", exc)
            return ParsedQuery(semantic_query=original_query)
