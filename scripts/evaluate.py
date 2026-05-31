#!/usr/bin/env python3
"""
Phase 8 — Offline evaluation against golden queries.

Computes NDCG@10 and Precision@5 for each golden query in
tests/golden_queries.json, then prints a comparison table.

Usage:
    python scripts/evaluate.py [--rerank] [--k 10]

The script does NOT require the Claude API — it runs queries directly
through the HybridRetriever + Scorer without the LLM parse and explain
stages (to keep evaluation deterministic and fast).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Optional

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.table import Table

console = Console()


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------

def _relevance(item_id: str, expected_top: list[str], expected_absent: list[str]) -> int:
    if item_id in expected_absent:
        return -1   # penalised (should not appear)
    if item_id in expected_top:
        return 2    # highly relevant
    return 0        # not judged


def ndcg_at_k(result_ids: list[str], golden: dict, k: int) -> float:
    top = golden["expected_top_ids"]
    absent = golden["expected_absent_ids"]
    rels = [max(0, _relevance(rid, top, absent)) for rid in result_ids[:k]]
    ideal = sorted(
        [max(0, _relevance(rid, top, absent)) for rid in top + result_ids[:k]],
        reverse=True,
    )[:k]

    def dcg(r: list[int]) -> float:
        return sum(v / math.log2(i + 2) for i, v in enumerate(r))

    idcg = dcg(ideal)
    return dcg(rels) / idcg if idcg > 0 else 0.0


def precision_at_k(result_ids: list[str], golden: dict, k: int) -> float:
    top = golden["expected_top_ids"]
    hits = sum(1 for rid in result_ids[:k] if rid in top)
    return hits / k


def penalty_score(result_ids: list[str], golden: dict, k: int) -> int:
    absent = golden["expected_absent_ids"]
    return sum(1 for rid in result_ids[:k] if rid in absent)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

@click.command()
@click.option("--k", default=10, show_default=True, help="Cut-off for NDCG and Precision.")
@click.option("--rerank", is_flag=True, default=False, help="Enable cross-encoder reranker.")
@click.option(
    "--golden",
    "golden_path",
    default=str(Path(__file__).parent.parent / "tests" / "golden_queries.json"),
    show_default=True,
    help="Path to golden queries JSON.",
)
def evaluate(k: int, rerank: bool, golden_path: str) -> None:
    """Run the retrieval pipeline against golden queries and report metrics."""
    from src.config import get_settings
    from src.embedder import Embedder
    from src.query_parser import QueryParser
    from src.retriever import HybridRetriever
    from src.scorer import Scorer
    from src.store import QdrantStore

    golden_queries = json.loads(Path(golden_path).read_text())
    cfg = get_settings()

    embedder = Embedder(cfg.embed_model)
    store = QdrantStore(cfg)
    retriever = HybridRetriever(store, embedder, cfg)
    scorer = Scorer(cfg)
    reranker = None
    if rerank:
        from src.reranker import Reranker
        reranker = Reranker(cfg)

    ndcg_scores: list[float] = []
    prec_scores: list[float] = []
    results_table: list[dict] = []

    console.rule("[bold cyan]Evaluation[/bold cyan]")

    for gq in golden_queries:
        query_text: str = gq["query"]
        console.print(f"  Running: [italic]{query_text!r}[/italic]…")

        from src.schema import ParsedQuery
        parsed = ParsedQuery(semantic_query=query_text)

        candidates = retriever.retrieve(parsed, top_k=min(k * 4, 200))
        if reranker:
            candidates = reranker.rerank(query_text, candidates, top_n=k * 2)
        ranked = scorer.score(candidates, parsed)[:k]

        result_ids = [r.item.id for r in ranked]
        ndcg = ndcg_at_k(result_ids, gq, k)
        prec = precision_at_k(result_ids, gq, k)
        pen = penalty_score(result_ids, gq, k)

        ndcg_scores.append(ndcg)
        prec_scores.append(prec)
        results_table.append({
            "query": query_text,
            "ndcg": ndcg,
            "prec": prec,
            "penalty": pen,
        })

    # Build results table
    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("Query", max_width=40)
    table.add_column(f"NDCG@{k}", width=9, justify="right")
    table.add_column(f"P@5", width=7, justify="right")
    table.add_column("Penalties", width=9, justify="right")

    for row in results_table:
        ndcg_color = "green" if row["ndcg"] >= 0.7 else ("yellow" if row["ndcg"] >= 0.4 else "red")
        table.add_row(
            row["query"][:38],
            f"[{ndcg_color}]{row['ndcg']:.3f}[/{ndcg_color}]",
            f"{row['prec']:.3f}",
            f"{'[red]' if row['penalty'] else ''}{row['penalty']}{'[/red]' if row['penalty'] else ''}",
        )

    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    mean_prec = sum(prec_scores) / len(prec_scores)
    table.add_section()
    table.add_row(
        "[bold]MEAN[/bold]",
        f"[bold]{mean_ndcg:.3f}[/bold]",
        f"[bold]{mean_prec:.3f}[/bold]",
        "",
    )

    console.print(table)
    console.print(
        f"\n[bold]Mean NDCG@{k}:[/bold] {mean_ndcg:.4f}   "
        f"[bold]Mean P@5:[/bold] {mean_prec:.4f}   "
        f"Queries: {len(golden_queries)}   Reranker: {'ON' if rerank else 'OFF'}"
    )


if __name__ == "__main__":
    evaluate()
