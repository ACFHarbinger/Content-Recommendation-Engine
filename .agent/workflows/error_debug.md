# Workflow — Error Debugging

1. **Reproduce** with the narrowest `recommend ...` invocation and the resolved config printed;
   save the failing prompt/dataset.
2. **Localize** along the pipeline: config → query parser → retrieval → fusion/scoring →
   explainer → output (per `.agent/prompts/debug.md`).
3. **Fix at the failing stage** (e.g. a wrong ranking is usually a config/fusion issue, not the LLM).
4. **Regression test**: add the failing case with the LLM/embedder mocked; extend
   `tests/golden_queries.json` if retrieval quality regressed.
5. **Verify**: `just test::test`, rerun the reproduction; note user-facing fixes in the CHANGELOG.
