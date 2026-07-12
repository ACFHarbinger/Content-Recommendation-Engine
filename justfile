# Recommendation-Engine — Root Justfile
# https://github.com/casey/just
#
# Recipes are organised into per-domain sub-modules under tools/. Invoke a
# sub-module recipe directly (e.g. `just run::query ...`, `just test::coverage`),
# or use the root shorthands below.

set shell := ["bash", "-c"]
set unstable := true

# --- Sub-module declarations (imported from tools/) ---

mod helper   "tools/helper/justfile"
mod dev      "tools/dev/justfile"
mod test     "tools/test/justfile"
mod quality  "tools/quality/justfile"
mod run      "tools/run/justfile"

# --- Default target ---

default: help

# List all commands across every sub-module
help:
    @just helper::help

# --- Setup & maintenance (→ tools/dev) ---

# Sync runtime + dev dependencies
sync:
    @just dev::sync

# Install pre-commit hooks
hooks:
    @just dev::hooks

# Update dependencies
update:
    @just dev::update

# Remove caches and build artifacts
clean:
    @just dev::clean

# --- Quality (→ tools/quality) ---

# Format all code
fmt:
    @just quality::fmt

# Lint (CI-equivalent) + mypy
lint:
    @just quality::lint

# Auto-fix lint issues
fix:
    @just quality::fix

# Type-check
typecheck:
    @just quality::typecheck

# Audit dependencies
audit:
    @just quality::audit

# --- Testing (→ tools/test) ---
# Note: the bare `test` name is the sub-module; use `just test::test` or the
# shorthand below.

# Run the full test suite
test-run:
    @just test::test

# Coverage report
coverage:
    @just test::coverage

# Quality gate: lint + tests
check: lint test-run
    @echo "✅ Code quality check passed!"

# --- Run (→ tools/run) ---

# Run the recommend CLI
cli *args:
    @just run::cli {{args}}

# Ingest a dataset
ingest *args:
    @just run::ingest {{args}}

# Query the engine
query *args:
    @just run::query {{args}}

# Evaluate retrieval quality
evaluate *args:
    @just run::evaluate {{args}}
