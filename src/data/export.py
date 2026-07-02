"""
Phase 1 — Export the SQLite store back to a JSON file.

Usage:
    python -m src.export --output data/backup.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
from rich.console import Console

from src.core.config import get_settings # pyrefly: ignore [missing-import]
from src.data.store import SQLiteStore # pyrefly: ignore [missing-import]

logger = logging.getLogger(__name__)
console = Console()


@click.command()
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(),
    help="Destination JSON file.",
)
@click.option(
    "--limit", default=10_000, show_default=True, help="Maximum items to export."
)
def export(output_path: str, limit: int) -> None:
    """Export the SQLite store back to a JSON file."""
    cfg = get_settings()
    store = SQLiteStore(cfg)

    console.rule("[bold cyan]Recommendation Engine — Export[/bold cyan]")

    with console.status("Fetching items from SQLite…"):
        rows = store.fetch_filtered(limit=limit)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    console.print(
        f"[bold green]✓[/bold green] Exported {len(rows)} items to [green]{out}[/green]"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    export()
