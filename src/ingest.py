"""
Phase 1 — Ingestion CLI entry point.

Usage:
    python -m src.ingest --input data/items.json [--batch-size 32] [--reset]

Validates each item against the MediaItem schema, embeds in batches with
BGE-M3, and upserts to Qdrant with a rich progress bar.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .config import get_settings
from .embedder import Embedder
from .schema import MediaItem
from .store import QdrantStore

logger = logging.getLogger(__name__)
console = Console()


def load_and_validate(path: Path) -> tuple[list[MediaItem], list[dict]]:
    """Parse JSON, validate with Pydantic.  Returns (valid_items, skipped_raw)."""
    with open(path, "r", encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    valid: list[MediaItem] = []
    skipped: list[dict] = []
    for i, item in enumerate(raw):
        try:
            valid.append(MediaItem.model_validate(item))
        except ValidationError as e:
            console.print(
                f"[yellow]⚠  Skipping item {i} (id={item.get('id','?')}): "
                f"{e.error_count()} validation error(s)[/yellow]"
            )
            skipped.append(item)

    return valid, skipped


@click.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True), help="Path to JSON file with media items.")
@click.option("--batch-size", "-b", default=32, show_default=True, help="Embedding batch size (reduce if OOM).")
@click.option("--reset", is_flag=True, default=False, help="Drop and recreate the Qdrant collection before ingesting.")
def ingest(input_path: str, batch_size: int, reset: bool) -> None:
    """Ingest a JSON media library into Qdrant."""
    cfg = get_settings()
    path = Path(input_path)

    console.rule("[bold cyan]Recommendation Engine — Ingest[/bold cyan]")
    console.print(f"Input    : [green]{path}[/green]")
    console.print(f"Collection: [green]{cfg.qdrant_collection}[/green]")
    console.print(f"Storage  : [green]{cfg.qdrant_storage_path or cfg.qdrant_url}[/green]")

    # Validate input
    with console.status("Validating input…"):
        items, skipped = load_and_validate(path)

    console.print(f"[green]✓[/green] Loaded {len(items)} valid items, skipped {len(skipped)}.")
    if not items:
        console.print("[red]No valid items to ingest.  Exiting.[/red]")
        sys.exit(1)

    # Store setup
    store = QdrantStore(cfg)
    if reset:
        try:
            store._get_client().delete_collection(cfg.qdrant_collection)
            console.print("[yellow]⚠  Dropped existing collection.[/yellow]")
        except Exception:
            pass
    store.create_collection()

    # Embed
    embedder = Embedder(model_name=cfg.embed_model, max_length=512)

    embedded = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Embedding…", total=len(items))

        def _cb(current: int, total: int) -> None:
            progress.update(task, completed=current)

        embedded = embedder.embed_batch(items, batch_size=batch_size, progress_callback=_cb)

    # Upsert
    with console.status(f"Upserting {len(embedded)} points to Qdrant…"):
        n = store.upsert(embedded, batch_size=64)

    info = store.collection_info()
    console.print(
        f"[bold green]✓ Done.[/bold green] "
        f"Upserted {n} points.  "
        f"Collection now has {info['points_count']} total points."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    ingest()
