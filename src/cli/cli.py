"""
CLI entry point for the Recommendation Engine.

Commands
--------
recommend ingest  --input FILE [--batch-size N] [--reset]
recommend export  --output FILE [--limit N]
recommend query   TEXT [--top-k N] [--format table|json] [--rerank] [--no-explain]
recommend info
"""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console

console = Console()


@click.group()
@click.option(
    "-v", "--verbose", is_flag=True, default=False, help="Enable DEBUG logging."
)
def cli(verbose: bool) -> None:
    """Personal media recommendation engine — local-first, BGE-M3 + SQLite."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(name)s: %(message)s",
    )


# ---- Phase 1 commands ------------------------------------------------

from src.data.ingest import ingest  # pyrefly: ignore [missing-import]
from src.data.export import export  # pyrefly: ignore [missing-import]

cli.add_command(ingest)
cli.add_command(export)


# ---- Phase 9 — sync --------------------------------------------------


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="JSON file with items to add or update.",
)
@click.option("--batch-size", "-b", default=32, show_default=True)
def sync(input_path: str, batch_size: int) -> None:
    """
    Incrementally upsert items from a JSON file.

    Items are matched by UUID — existing items are updated in-place,
    new items are added.  Items not in the file are left untouched.
    """
    import json
    from pathlib import Path
    from pydantic import ValidationError
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from src.core.config import get_settings # pyrefly: ignore [missing-import]
    from src.data.embedder import Embedder # pyrefly: ignore [missing-import]
    from src.core.schema import MediaItem # pyrefly: ignore [missing-import]
    from src.data.store import SQLiteStore # pyrefly: ignore [missing-import]

    cfg = get_settings()
    path = Path(input_path)

    with console.status("Validating input…"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        items, skipped = [], []
        for i, rec in enumerate(raw):
            try:
                items.append(MediaItem.model_validate(rec))
            except ValidationError:
                skipped.append(i)

    console.print(f"[green]✓[/green] {len(items)} valid, {len(skipped)} skipped.")

    if not items:
        console.print("[yellow]Nothing to sync.[/yellow]")
        return

    embedder = Embedder(cfg.embed_model)
    store = SQLiteStore(cfg)
    store.create_collection()  # idempotent

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as prog:
        task = prog.add_task("Embedding…", total=len(items))

        def _cb(cur, tot):
            prog.update(task, completed=cur)

        embedded = embedder.embed_batch(
            items, batch_size=batch_size, progress_callback=_cb
        )

    with console.status(f"Upserting {len(embedded)} items…"):
        n = store.upsert(embedded)

    info = store.collection_info()
    console.print(
        f"[bold green]✓[/bold green] Synced {n} items. Store: {info['points_count']} total."
    )


# ---- Phase 9 — delete ------------------------------------------------


@cli.command()
@click.argument("item_id")
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt."
)
def delete(item_id: str, yes: bool) -> None:
    """Remove a single item from the store by UUID."""
    from src.core.config import get_settings # pyrefly: ignore [missing-import]
    from src.data.store import SQLiteStore # pyrefly: ignore [missing-import]

    if not yes:
        click.confirm(f"Delete item {item_id!r} from the store?", abort=True)

    cfg = get_settings()
    store = SQLiteStore(cfg)
    store.delete(item_id)
    console.print(f"[green]✓[/green] Deleted {item_id!r}.")


# ---- Phase 6 — info --------------------------------------------------


@cli.command()
def info() -> None:
    """Show store statistics and configuration."""
    from src.core.config import get_settings # pyrefly: ignore [missing-import]
    from src.data.store import SQLiteStore # pyrefly: ignore [missing-import]

    cfg = get_settings()
    store = SQLiteStore(cfg)
    try:
        data = store.collection_info()
        console.print(f"[bold]Storage     :[/bold] {data['storage']}")
        console.print(f"[bold]Items       :[/bold] {data['points_count']}")
        console.print(f"[bold]Embed model :[/bold] {cfg.embed_model}")
        console.print(f"[bold]Claude model:[/bold] {cfg.claude_model}")
        console.print(f"[bold]Fusion      :[/bold] {cfg.fusion_method.upper()}")
        console.print(f"[bold]λ recency   :[/bold] {cfg.lambda_recency}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


# ---- Phase 6 — query -------------------------------------------------


@cli.command()
@click.argument("query_text")
@click.option(
    "--top-k",
    "-k",
    default=None,
    type=int,
    help="Number of results to return (default: DEFAULT_TOP_K from config).",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--rerank",
    is_flag=True,
    default=False,
    help="Apply cross-encoder reranker (Phase 8, slower but more precise).",
)
@click.option(
    "--no-explain",
    is_flag=True,
    default=False,
    help="Skip LLM explanation generation (faster).",
)
@click.option(
    "--no-history",
    is_flag=True,
    default=False,
    help="Disable watch-history implicit feedback boost.",
)
def query(
    query_text: str,
    top_k: int | None,
    fmt: str,
    rerank: bool,
    no_explain: bool,
    no_history: bool,
) -> None:
    """Run the full recommendation pipeline for QUERY_TEXT."""
    from src.core.config import get_settings # pyrefly: ignore [missing-import]
    from src.search.pipeline import RecommendationPipeline # pyrefly: ignore [missing-import]
    from src.cli.output import print_table, to_json # pyrefly: ignore [missing-import]

    cfg = get_settings()

    with console.status("[cyan]Running recommendation pipeline…[/cyan]"):
        pipeline = RecommendationPipeline(
            top_k=top_k,
            use_reranker=rerank,
            explain=not no_explain,
            use_history=not no_history,
            settings=cfg,
        )
        results = pipeline.run(query_text)

    if not results:
        console.print("[yellow]No results found for this query.[/yellow]")
        return

    if fmt == "json":
        print(to_json(results))
    else:
        console.rule(f"[bold cyan]Results for: {query_text!r}[/bold cyan]")
        print_table(results)


if __name__ == "__main__":
    cli()
