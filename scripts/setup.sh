#!/usr/bin/env bash
# One command to make a fresh clone usable. Run it once per machine.
#
#   bash scripts/setup.sh
#
# Git never runs anything on clone (that would be arbitrary code execution from
# whoever wrote the repo), so hooks cannot self-install. This is that one step.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "==> syncing environment"
uv sync

echo "==> installing git hooks"
# pre-commit refuses to install while core.hooksPath is set.
git config --unset core.hooksPath 2>/dev/null || true
.venv/bin/pre-commit install

echo
echo "ready."
echo "note: datasets/ and derived/ are not in git — copy them over by hand."
