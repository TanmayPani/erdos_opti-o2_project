#!/usr/bin/env bash
# Export slides.py -> site/index.html, driven by .pre-commit-config.yaml.
#
# Records the index blob hashes of every source the deck is built from in
# site/.build-manifest. If they already match, the export is skipped instantly —
# which is what makes pre-commit's "files were modified by this hook" re-run
# cost a second instead of another full build.
#
# Skip entirely: git commit --no-verify
set -uo pipefail

EXPORT_TIMEOUT=300   # measured ~20s; 15x headroom, matches AGENTS.md

SOURCES=(slides.py eda.py utils.py core assets pyproject.toml scripts/export-slides.sh)

cd "$(git rev-parse --show-toplevel)" || exit 1

say()  { printf '\033[36mdeck\033[0m %s\n' "$1" >&2; }
fail() { printf '\033[31mdeck\033[0m %s\n' "$1" >&2; exit 1; }

# Index blob hashes, not working-tree ones: the manifest must describe exactly
# what this commit will contain. pre-commit stashes unstaged changes before
# running hooks, so on-disk == index while the export runs; the two agree.
manifest() { git ls-files -s -- "${SOURCES[@]}" | awk '{print $2"  "$4}' | sort; }

mkdir -p site
want="$(manifest)"
[ -n "$want" ] || fail "no deck sources found in the index"

if [ -f site/.build-manifest ] && [ -f site/index.html ] \
   && [ "$want" = "$(cat site/.build-manifest)" ]; then
  say "site/index.html already current — nothing to build"
  exit 0
fi

tmp="$(mktemp "$PWD/site/.export.XXXXXX.html")"
log="$(mktemp)"
trap 'rm -f "$tmp" "$log"' EXIT

say "exporting slides.py (~20s)..."
timeout "$EXPORT_TIMEOUT" .venv/bin/marimo export html slides.py --no-include-code -o "$tmp" >"$log" 2>&1
rc=$?

[ "$rc" -eq 124 ] && { tail -20 "$log" | sed 's/^/  | /' >&2; fail "timed out after ${EXPORT_TIMEOUT}s"; }
[ "$rc" -ne 0 ]   && { tail -20 "$log" | sed 's/^/  | /' >&2; fail "marimo exited $rc"; }

# marimo exits 0 even when cells raise, so the log is the real check.
if grep -qE 'MarimoExceptionRaisedError|^Traceback' "$log"; then
  grep -nE 'MarimoExceptionRaisedError|^Traceback' "$log" | head -5 | sed 's/^/  | /' >&2
  fail "cells raised during export — the deck would be incomplete"
fi
grep -q "Your output is too large" "$tmp" && fail "an output hit marimo's 8 MB cap and was dropped"
[ -s "$tmp" ] || fail "export produced an empty file"

mv "$tmp" site/index.html
touch site/.nojekyll
printf '%s\n' "$want" > site/.build-manifest
git add site/index.html site/.nojekyll site/.build-manifest
say "built site/index.html ($(( $(stat -c %s site/index.html) / 1000000 )) MB)"
exit 0
