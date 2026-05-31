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
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable DEBUG logging.")
def cli(verbose: bool) -> None:
    """Personal media recommendation engine — local-first, BGE-M3 + Qdrant."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(name)s: %(message)s",
    )


# ---- Phase 1 commands ------------------------------------------------

from .ingest import ingest   # noqa: E402
from .export import export   # noqa: E402

cli.add_command(ingest)
cli.add_command(export)


# ---- Phase 6 — info --------------------------------------------------

@cli.command()
def info() -> None:
    """Show collection statistics and configuration."""
    from .config import get_settings
    from .store import QdrantStore

    cfg = get_settings()
    store = QdrantStore(cfg)
    try:
        data = store.collection_info()
        console.print(f"[bold]Collection  :[/bold] {data['collection']}")
        console.print(f"[bold]Points      :[/bold] {data['points_count']}")
        console.print(f"[bold]Storage     :[/bold] {data['storage']}")
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
    "--top-k", "-k", default=None, type=int,
    help="Number of results to return (default: DEFAULT_TOP_K from config).",
)
@click.option(
    "--format", "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--rerank", is_flag=True, default=False,
    help="Apply cross-encoder reranker (Phase 8, slower but more precise).",
)
@click.option(
    "--no-explain", is_flag=True, default=False,
    help="Skip LLM explanation generation (faster).",
)
def query(
    query_text: str,
    top_k: int | None,
    fmt: str,
    rerank: bool,
    no_explain: bool,
) -> None:
    """Run the full recommendation pipeline for QUERY_TEXT."""
    from .config import get_settings
    from .pipeline import RecommendationPipeline
    from .output import print_table, to_json

    cfg = get_settings()

    with console.status(f"[cyan]Running recommendation pipeline…[/cyan]"):
        pipeline = RecommendationPipeline(
            top_k=top_k,
            use_reranker=rerank,
            explain=not no_explain,
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
