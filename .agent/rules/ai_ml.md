# Rules — AI / Retrieval

- **LLM-optional**: the Anthropic parser and explainer are enhancements, not hard dependencies.
  Always provide a deterministic fallback and degrade gracefully without `ANTHROPIC_API_KEY`.
- **Explainability**: the explainer may only cite tags/fields that actually exist on the item —
  no hallucinated reasons. Validate against the item before returning.
- **Determinism in tests**: mock the LLM and the embedder; assert on structure (non-empty ranked
  list, reasons reference real tags, scores in range) — never on exact LLM text or float equality.
- **Prompt caching**: keep system prompts stable (query parser + explainer use prompt caching);
  don't reorder them casually.
- **Retrieval math**: guard empty-candidate / NaN cases in cosine/dot/RRF/DBSF; document the
  candidate cap and fusion weights (all config-driven).
- **Embeddings**: BGE-M3 dense = semantic (reviews/notes), sparse = lexical (title/tags/genres/
  entities); batch + cache; document CPU/VRAM footprint.
