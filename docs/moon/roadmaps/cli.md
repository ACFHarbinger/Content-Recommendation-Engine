# Roadmap — CLI & Output

The user-facing layer. Implementation in [`src/cli/`](../../src/cli/) (`cli.py`, `output.py`),
entry point `recommend`.

## §1 — CLI (Click)

- [x] `recommend` command group: ingest and query flows.
- [ ] `--help` documents every option; consistent exit codes; `--json` on every query command.

## §2 — Output (rich)

- [x] Rich terminal rendering of ranked results with reasons + Recommendation Value.
- [ ] Machine-readable JSON output mirrors the terminal fields exactly (for scripting / Image-Toolkit).

## §3 — Integration

- [ ] Document how the parent Image-Toolkit invokes `recommend` (entry point + JSON contract).
