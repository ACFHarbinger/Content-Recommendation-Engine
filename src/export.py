"""
Phase 1 — Export current Qdrant collection back to JSON.

Usage:
    python -m src.export --output data/backup.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import click
from rich.console import Console

from .config import get_settings
from .store import QdrantStore

logger = logging.getLogger(__name__)
console = Console()


@click.command()
@click.option("--output", "-o", "output_path", required=True, type=click.Path(), help="Destination JSON file.")
@click.option("--limit", default=10_000, show_default=True, help="Maximum points to export.")
def export(output_path: str, limit: int) -> None:
    """Export the Qdrant collection back to a JSON file."""
    cfg = get_settings()
    store = QdrantStore(cfg)
    client = store._get_client()

    console.rule("[bold cyan]Recommendation Engine — Export[/bold cyan]")

    records = []
    offset = None
    with console.status("Fetching points from Qdrant…"):
        while len(records) < limit:
            batch, next_offset = client.scroll(
                collection_name=cfg.qdrant_collection,
                limit=min(256, limit - len(records)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(p.payload for p in batch if p.payload)
            if next_offset is None:
                break
            offset = next_offset

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    console.print(
        f"[bold green]✓[/bold green] Exported {len(records)} items to [green]{out}[/green]"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    export()
