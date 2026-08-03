# Workflow — Refactoring

1. **Contract first**: write down the public surface (CLI commands + `--json` shape, config keys,
   schema fields) that must not change.
2. **Net**: ensure the area is covered; add characterization tests if not.
3. **Mechanical vs behavioural** in separate commits; renamed config keys keep a deprecation alias
   for one release.
4. **Gate**: `just lint && just typecheck && just test::test`.
5. **Docs**: module docstrings, `docs/ARCHITECTURE.md` map, affected roadmap, CHANGELOG.
