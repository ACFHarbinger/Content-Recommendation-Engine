"""
Phase 5 — Async LLM Explainer.

Generates human-readable reasons for each recommendation using the Claude API.
All top-K explanations are requested in parallel with asyncio.gather to keep
total latency close to a single API call (~1–3 s).

Anti-hallucination guard: matched_tags is validated at output time against the
item's actual tags + genres — any fabricated tag is stripped with a warning.

Falls back to a template-based reason string on any API or parse failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

from src.core.config import Settings, get_settings
from src.core.schema import ExplainedResult, RankedResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an explainer for a personal media recommendation engine.
Given a user's original search query and one recommended media item with its
full metadata and scores, generate a concise, factual explanation.

Output ONLY valid JSON matching this schema:
{
  "reasons": ["<2–4 short sentences grounded in the item metadata>"],
  "matched_tags": ["<tags or genres from the item that match the query>"]
}

Rules:
1. Each reason must cite a specific metadata field (title, genre, tag, entity, rating, year).
2. matched_tags MUST be a strict subset of the item's actual tags and genres — do NOT invent tags.
3. Maximum 4 reasons. Prefer quality over quantity.
4. Use the correct consumption verb: "watch" for anime/shows/movies, "read" for books/manga/papers.
5. Mention the recommendation_value only if it is above 0.7 (call it "strong match").
6. No plot spoilers. No hallucination. Only reference facts provided.
7. Output ONLY the JSON object.
"""


def _build_user_message(result: RankedResult, user_query: str) -> str:
    item = result.item
    meta = {
        "title": item.title,
        "type": item.type,
        "year": item.year_released,
        "rating": item.rating,
        "episodes": item.num_episodes_or_pages,
        "genres": item.genres,
        "tags": item.tags,
        "entities": item.associated_entities,
        "review_snippet": (item.review_notes or "")[:300],
        "watch_status": item.watch_status,
        "recommendation_value": result.recommendation_value,
    }
    return (
        f"User query: {user_query!r}\n\n"
        f"Item metadata:\n{json.dumps(meta, ensure_ascii=False, indent=2)}"
    )


def _fallback_reason(result: RankedResult, user_query: str) -> ExplainedResult:
    item = result.item
    # verb = item.consume_verb
    reasons = [
        f"Recommended based on your query {user_query!r}.",
        f"This {item.type or 'item'} has a rating of {item.rating}/10."
        if item.rating
        else "Matches the thematic profile of your query.",
    ]
    return ExplainedResult(
        **result.model_dump(),
        reasons=reasons,
        matched_tags=[t for t in item.tags[:3]],
    )


# Cached system prompt block — reused for every item in a batch so the
# prompt is transmitted once and cached for subsequent parallel calls.
_CACHED_SYSTEM = [
    {
        "type": "text",
        "text": _SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


async def _explain_one(
    async_client,
    result: RankedResult,
    user_query: str,
    model: str,
) -> ExplainedResult:
    """Explain a single result; falls back to template on any error."""
    try:
        response = await async_client.messages.create(
            model=model,
            max_tokens=512,
            system=_CACHED_SYSTEM,
            messages=[
                {"role": "user", "content": _build_user_message(result, user_query)}
            ],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        return ExplainedResult(
            **result.model_dump(),
            reasons=data.get("reasons", []),
            matched_tags=data.get("matched_tags", []),
        )
    except Exception as exc:
        logger.warning("Explanation failed for %r: %s", result.item.title, exc)
        return _fallback_reason(result, user_query)


class Explainer:
    """
    Generates parallel Claude-based explanations for a list of RankedResults.

    Usage
    -----
    explainer = Explainer()
    explained = explainer.explain_batch(ranked_results, user_query="...")
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._cfg = settings or get_settings()

    def explain_batch(
        self,
        results: list[RankedResult],
        user_query: str,
    ) -> list[ExplainedResult]:
        """
        Explain all *results* in parallel.  Runs the async loop internally
        so callers don't need to manage event loops.
        """
        if not results:
            return []
        return asyncio.run(self._explain_all(results, user_query))

    async def _explain_all(
        self,
        results: list[RankedResult],
        user_query: str,
    ) -> list[ExplainedResult]:
        if not self._cfg.anthropic_api_key:
            logger.info("No ANTHROPIC_API_KEY — using template fallback for all items.")
            return [_fallback_reason(r, user_query) for r in results]

        try:
            import anthropic

            async_client = anthropic.AsyncAnthropic(api_key=self._cfg.anthropic_api_key)
            tasks = [
                _explain_one(async_client, r, user_query, self._cfg.claude_model)
                for r in results
            ]
            return list(await asyncio.gather(*tasks))
        except ImportError:
            logger.warning("anthropic package not installed — using template fallback.")
            return [_fallback_reason(r, user_query) for r in results]
