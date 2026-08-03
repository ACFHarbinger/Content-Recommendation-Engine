#!/usr/bin/env python3
"""
Phase 8 / Phase 9 — Hyperparameter sweep over scoring decay parameters.

Runs a grid search over combinations of:
  - lambda_recency  ∈ [0.02, 0.05, 0.1]
  - fusion_method   ∈ ["rrf", "dbsf"]

for each combination, evaluates NDCG@K on the golden query set and
reports a comparison table so you can pick the best config.

Usage:
    python scripts/sweep.py [--k 10] [--csv sweep_results.csv]

Requires a populated Qdrant collection:
    recommend ingest --input data/sample.json
    recommend ingest --input data/sample_books.json  # optional
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.table import Table

console = Console()

# Grid dimensions
LAMBDA_RECENCY_VALUES = [0.02, 0.05, 0.10]
FUSION_METHODS = ["rrf", "dbsf"]


# ------------------------------------------------------------------
# Metric helpers (shared with evaluate.py)
# ------------------------------------------------------------------


def _relevance(
    item_id: str, expected_top: list[str], expected_absent: list[str]
) -> int:
    if item_id in expected_absent:
        return -1
    if item_id in expected_top:
        return 2
    return 0


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


def _evaluate_config(
    lambda_recency: float,
    fusion_method: str,
    golden_queries: list[dict],
    embedder,
    store,
    k: int,
) -> float:
    """Run all golden queries with given hyperparams; return mean NDCG@K."""
    from src.config import Settings  # pyrefly: ignore [missing-import]
    from src.query_parser import _build_qdrant_filter  # pyrefly: ignore [missing-import]
    from src.retriever import HybridRetriever  # pyrefly: ignore [missing-import]
    from src.schema import FilterClause, ParsedQuery  # pyrefly: ignore [missing-import]
    from src.scorer import Scorer  # pyrefly: ignore [missing-import]

    cfg = Settings(
        qdrant_local_path=store._cfg.qdrant_local_path,
        lambda_recency=lambda_recency,
        fusion_method=fusion_method,
    )
    retriever = HybridRetriever(store, embedder, cfg)
    scorer = Scorer(cfg)

    ndcg_scores = []
    for gq in golden_queries:
        raw_filters = gq.get("parsed_filters", [])
        filter_clauses = [FilterClause(**f) for f in raw_filters]
        parsed = ParsedQuery(
            semantic_query=gq.get("semantic_query", gq["query"]),
            filters=filter_clauses,
        )
        qdrant_filter = _build_qdrant_filter(filter_clauses)
        candidates = retriever.retrieve(
            parsed, top_k=min(k * 4, 200), qdrant_filter=qdrant_filter
        )
        ranked = scorer.score(candidates, parsed)[:k]
        result_ids = [r.item.id for r in ranked]
        ndcg_scores.append(ndcg_at_k(result_ids, gq, k))

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


@click.command()
@click.option("--k", default=10, show_default=True)
@click.option(
    "--golden",
    "golden_path",
    default=str(Path(__file__).parent.parent / "tests" / "golden_queries.json"),
    show_default=True,
)
@click.option("--csv", "csv_path", default=None, help="Write results to this CSV.")
def sweep(k: int, golden_path: str, csv_path: Optional[str]) -> None:
    """
    Grid search over lambda_recency × fusion_method.
    Prints NDCG@K for every combination and marks the best.
    """
    from src.config import get_settings  # pyrefly: ignore [missing-import]
    from src.embedder import Embedder  # pyrefly: ignore [missing-import]
    from src.store import QdrantStore  # pyrefly: ignore [missing-import]

    golden_queries = json.loads(Path(golden_path).read_text())
    cfg = get_settings()
    embedder = Embedder(cfg.embed_model)
    store = QdrantStore(cfg)

    console.rule("[bold cyan]Hyperparameter Sweep[/bold cyan]")
    console.print(f"  Grid: λ ∈ {LAMBDA_RECENCY_VALUES}  ×  fusion ∈ {FUSION_METHODS}")
    console.print(f"  Queries: {len(golden_queries)}   K={k}\n")

    combinations = list(itertools.product(LAMBDA_RECENCY_VALUES, FUSION_METHODS))
    results: list[dict] = []

    for lam, fusion in combinations:
        label = f"λ={lam:.2f}  fusion={fusion}"
        with console.status(f"  Running {label}…"):
            score = _evaluate_config(lam, fusion, golden_queries, embedder, store, k)
        results.append(
            {
                "lambda_recency": lam,
                "fusion_method": fusion,
                "mean_ndcg": round(score, 4),
            }
        )
        console.print(f"  {label:30s}  NDCG@{k} = {score:.4f}")

    best = max(results, key=lambda r: r["mean_ndcg"])

    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("λ recency", justify="center", width=12)
    table.add_column("Fusion", justify="center", width=10)
    table.add_column(f"NDCG@{k}", justify="right", width=10)
    table.add_column("", width=6)

    for r in sorted(results, key=lambda x: x["mean_ndcg"], reverse=True):
        is_best = r is best
        tag = "[bold green]★ best[/bold green]" if is_best else ""
        color = "green" if is_best else ("yellow" if r["mean_ndcg"] >= 0.4 else "red")
        table.add_row(
            str(r["lambda_recency"]),
            r["fusion_method"].upper(),
            f"[{color}]{r['mean_ndcg']:.4f}[/{color}]",
            tag,
        )

    console.print()
    console.print(table)
    console.print(
        f"\n[bold green]Best config:[/bold green] "
        f"λ_recency={best['lambda_recency']}  "
        f"fusion_method={best['fusion_method']}  "
        f"NDCG@{k}={best['mean_ndcg']:.4f}"
    )

    if csv_path:
        out = Path(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["lambda_recency", "fusion_method", "mean_ndcg"]
            )
            writer.writeheader()
            writer.writerows(results)
        console.print(f"[dim]Sweep results saved to {out}[/dim]")


if __name__ == "__main__":
    sweep()
