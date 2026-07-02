"""
Phase 6 — Output renderers.

Supports two formats:
  table  — Rich-formatted console table (default)
  json   — JSON-serialisable list of dicts

Both renderers handle the full ExplainedResult including reasons,
matched_tags, component scores, and media links.
"""

from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.core.schema import ExplainedResult # pyrefly: ignore [missing-import]

console = Console()


# ------------------------------------------------------------------
# JSON output
# ------------------------------------------------------------------


def to_json(results: list[ExplainedResult], indent: int = 2) -> str:
    """Serialise results to a JSON string."""
    rows = []
    for r in results:
        rows.append(
            {
                "rank": r.rank,
                "title": r.item.title,
                "type": r.item.type,
                "year": r.item.year_released,
                "rating": r.item.rating,
                "watch_status": r.item.watch_status,
                "recommendation_value": r.recommendation_value,
                "component_scores": r.component_scores.model_dump(),
                "reasons": r.reasons,
                "matched_tags": r.matched_tags,
                "genres": r.item.genres,
                "tags": r.item.tags,
                "web_link": r.item.web_link,
                "local_file": r.item.local_file_location,
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=indent)


# ------------------------------------------------------------------
# Rich table output
# ------------------------------------------------------------------

_TYPE_COLORS = {
    "anime": "cyan",
    "movie": "magenta",
    "show": "blue",
    "book": "green",
    "manga": "yellow",
    "paper": "white",
    "game": "orange3",
}


def _type_badge(t: str | None) -> Text:
    t = (t or "other").lower()
    color = _TYPE_COLORS.get(t, "dim white")
    return Text(t.upper(), style=f"bold {color}")


def _score_bar(value: float, width: int = 10) -> str:
    filled = max(0, min(width, round(value * width)))
    return "█" * filled + "░" * (width - filled)


def print_table(results: list[ExplainedResult]) -> None:
    """Render results to the console as a Rich table."""
    if not results:
        console.print("[yellow]No results.[/yellow]")
        return

    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        expand=False,
        leading=1,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", min_width=22, max_width=36)
    table.add_column("Type", width=7)
    table.add_column("Year", width=5)
    table.add_column("⭐", width=5)
    table.add_column("Score", width=13)
    table.add_column("Reasons", min_width=40)

    for r in results:
        item = r.item
        rec_val = r.recommendation_value
        bar = _score_bar(min(rec_val / 1.5, 1.0))
        score_cell = f"[bold]{rec_val:.3f}[/bold]\n[dim]{bar}[/dim]"

        rating_str = f"{item.rating:.1f}" if item.rating else "—"
        year_str = str(item.year_released) if item.year_released else "—"

        reason_lines = (
            "\n".join(f"• {r_}" for r_ in r.reasons[:3]) if r.reasons else "—"
        )
        if r.matched_tags:
            reason_lines += "\n[dim]Tags: " + ", ".join(r.matched_tags[:5]) + "[/dim]"

        links = []
        if item.web_link:
            links.append(f"[dim][link]{item.web_link}[/link] web[/dim]")
        if item.local_file_location:
            links.append("[dim]📁 local[/dim]")
        if links:
            reason_lines += "\n" + "  ".join(links)

        table.add_row(
            str(r.rank),
            f"[bold]{item.title}[/bold]",
            _type_badge(item.type),
            year_str,
            rating_str,
            score_cell,
            reason_lines,
        )

    console.print(table)
    console.print(
        f"[dim]Showing {len(results)} result(s) · "
        "RecVal = rrf_score × rating_boost × recency_decay × length_decay[/dim]"
    )
