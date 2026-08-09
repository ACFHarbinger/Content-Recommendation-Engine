# Debugging Prompt

```
Debug the following CRE failure: {SYMPTOM}.

Triage order (matches the pipeline):
1. Config — is the setting resolved as expected? (print the pydantic settings; a wrong path or
   weight explains most "wrong results").
2. Query parser — did the LLM produce a sane {semantic_query, SQL WHERE}? Log the parsed query;
   check the deterministic fallback when no API key is set.
3. Retrieval — dense/sparse vectors present for the candidates? SQL WHERE filtering as intended?
4. Fusion/scoring — RRF/DBSF + decay math; check for NaN/empty-candidate edge cases.
5. Explainer — hallucinated reasons? It must cite only fields/tags present on the item.
6. Output — CLI vs JSON parity.

Reproduce with the narrowest `recommend ...` invocation, mock the LLM/embedder, add a regression
test, and fix at the failing stage.
```
