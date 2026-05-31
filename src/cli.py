"""
Phase 1 / Phase 6 — Click CLI group.

Current commands (Phase 0–1):
    recommend ingest  --input FILE
    recommend export  --output FILE
    recommend info

Stub commands (wired in later phases):
    recommend query   TEXT  (Phase 6)
"""
from __future__ import annotations

import click
from rich.console import Console

console = Console()


@click.group()
def cli() -> None:
    """Personal media recommendation engine — local-first, BGE-M3 + Qdrant."""


# ---- Phase 1 commands ------------------------------------------------

from .ingest import ingest  # noqa: E402
from .export import export  # noqa: E402

cli.add_command(ingest)
cli.add_command(export)


@cli.command()
def info() -> None:
    """Show collection statistics."""
    from .config import get_settings
    from .store import QdrantStore

    cfg = get_settings()
    store = QdrantStore(cfg)
    try:
        data = store.collection_info()
        console.print(f"[bold]Collection:[/bold] {data['collection']}")
        console.print(f"[bold]Points    :[/bold] {data['points_count']}")
        console.print(f"[bold]Storage   :[/bold] {data['storage']}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


# ---- Phase 6 stub ----------------------------------------------------

@cli.command()
@click.argument("query_text")
@click.option("--top-k", "-k", default=10, show_default=True)
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table", show_default=True)
def query(query_text: str, top_k: int, fmt: str) -> None:
    """Query the recommendation engine (Phase 6 — not yet implemented)."""
    console.print(
        "[yellow]⚠  The 'query' command requires Phases 2–6 to be implemented.[/yellow]"
    )
    console.print(
        f"Received query: [italic]{query_text!r}[/italic] (top_k={top_k}, format={fmt})"
    )


if __name__ == "__main__":
    cli()
