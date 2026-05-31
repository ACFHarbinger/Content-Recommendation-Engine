#!/usr/bin/env python3
"""
Phase 8 / Phase 9 — Offline evaluation against golden queries.

Computes NDCG@K and Precision@5 for each golden query in
tests/golden_queries.json.  Requires a populated Qdrant collection
(run ``recommend ingest`` first).

Usage:
    python scripts/evaluate.py [--k 10] [--rerank] [--csv results.csv]

Each golden query carries pre-parsed ``semantic_query`` and
``parsed_filters`` so evaluation is deterministic and does not require
a Claude API call.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional

# Make src importable from scripts/
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
        return -1   # penalised — should not appear
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
    return hits / k if k else 0.0


def penalty_score(result_ids: list[str], golden: dict, k: int) -> int:
    absent = golden["expected_absent_ids"]
    return sum(1 for rid in result_ids[:k] if rid in absent)


# ------------------------------------------------------------------
# Query runner (uses pre-parsed filters — no Claude dependency)
# ------------------------------------------------------------------

def _run_golden_query(
    gq: dict,
    retriever,
    scorer,
    reranker,
    top_k: int,
) -> list[str]:
    """Run a single golden query and return the result UUIDs."""
    from src.query_parser import _build_qdrant_filter
    from src.schema import FilterClause, ParsedQuery

    raw_filters = gq.get("parsed_filters", [])
    filter_clauses = [FilterClause(**f) for f in raw_filters]
    parsed = ParsedQuery(
        semantic_query=gq.get("semantic_query", gq["query"]),
        filters=filter_clauses,
        raw_filters=raw_filters,
    )
    qdrant_filter = _build_qdrant_filter(filter_clauses)

    candidates = retriever.retrieve(parsed, top_k=min(top_k * 4, 200), qdrant_filter=qdrant_filter)

    if reranker:
        candidates = reranker.rerank(
            query=parsed.semantic_query or gq["query"],
            candidates=candidates,
            top_n=top_k * 2,
        )

    ranked = scorer.score(candidates, parsed)[:top_k]
    return [r.item.id for r in ranked]


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
@click.option(
    "--csv",
    "csv_path",
    default=None,
    help="Write per-query metrics to this CSV file.",
)
def evaluate(k: int, rerank: bool, golden_path: str, csv_path: Optional[str]) -> None:
    """Evaluate the retrieval pipeline against the golden query set."""
    from src.config import get_settings
    from src.embedder import Embedder
    from src.retriever import HybridRetriever
    from src.scorer import Scorer
    from src.store import QdrantStore

    golden_queries = json.loads(Path(golden_path).read_text())
    cfg = get_settings()

    embedder = Embedder(cfg.embed_model)
    store = QdrantStore(cfg)
    retriever = HybridRetriever(store, embedder, cfg)
    scorer = Scorer(cfg)
    reranker_obj = None
    if rerank:
        from src.reranker import Reranker
        reranker_obj = Reranker(cfg)

    ndcg_scores: list[float] = []
    prec_scores: list[float] = []
    rows: list[dict] = []

    console.rule("[bold cyan]Evaluation[/bold cyan]")
    console.print(f"  Golden queries : {len(golden_queries)}")
    console.print(f"  NDCG@{k} / P@5  : computing…\n")

    for gq in golden_queries:
        console.print(f"  [dim]{gq['query']!r}[/dim]")
        result_ids = _run_golden_query(gq, retriever, scorer, reranker_obj, k)
        ndcg = ndcg_at_k(result_ids, gq, k)
        prec = precision_at_k(result_ids, gq, min(5, k))
        pen = penalty_score(result_ids, gq, k)

        ndcg_scores.append(ndcg)
        prec_scores.append(prec)
        rows.append({
            "query": gq["query"],
            "ndcg": round(ndcg, 4),
            "precision_at_5": round(prec, 4),
            "penalties": pen,
            "result_ids": "|".join(result_ids),
        })

    # Rich table
    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("Query", max_width=42)
    table.add_column(f"NDCG@{k}", width=9, justify="right")
    table.add_column("P@5", width=7, justify="right")
    table.add_column("Pen", width=5, justify="right")

    for row in rows:
        color = "green" if row["ndcg"] >= 0.7 else ("yellow" if row["ndcg"] >= 0.4 else "red")
        table.add_row(
            row["query"][:40],
            f"[{color}]{row['ndcg']:.4f}[/{color}]",
            f"{row['precision_at_5']:.4f}",
            str(row["penalties"]) if row["penalties"] else "[dim]0[/dim]",
        )

    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    mean_prec = sum(prec_scores) / len(prec_scores)
    table.add_section()
    table.add_row(
        "[bold]MEAN[/bold]",
        f"[bold]{mean_ndcg:.4f}[/bold]",
        f"[bold]{mean_prec:.4f}[/bold]",
        "",
    )

    console.print()
    console.print(table)
    console.print(
        f"\n[bold]Mean NDCG@{k}:[/bold] {mean_ndcg:.4f}   "
        f"[bold]Mean P@5:[/bold] {mean_prec:.4f}   "
        f"Queries: {len(golden_queries)}   "
        f"Reranker: {'ON' if rerank else 'OFF'}"
    )

    # Optional CSV export
    if csv_path:
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"\n[dim]Results saved to {out}[/dim]")


if __name__ == "__main__":
    evaluate()
