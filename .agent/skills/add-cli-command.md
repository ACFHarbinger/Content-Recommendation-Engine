# Skill: Add a CLI Command

Add a subcommand to the `recommend` CLI (`src/cli/cli.py`, entry point `recommend = src.cli.cli:cli`).

1. **Click command**: add under the existing group; every option documented in `--help`; consistent
   exit codes.
2. **JSON parity**: if it returns results, support `--json` with fields mirroring the rich output
   (`src/cli/output.py`) — Image-Toolkit consumes the JSON contract.
3. **Thin layer**: the command orchestrates `src/search`/`src/data`; no business logic in the CLI.
4. **Config**: read tunables from `src/core/config.py`; secrets from env.
5. **Tests**: use Click's `CliRunner`; mock LLM/embedder; assert on exit code + JSON shape.
6. **Docs**: CLI-commands section of the README, `moon/roadmaps/cli.md`, CHANGELOG.
