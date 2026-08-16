#!/usr/bin/env bash
# Install the deep-think plugin into your Hermes home.
#
# Usage: ./install.sh [--hermes-home DIR] [--symlink]
#
# - Copies plugin/ → $HERMES_HOME/plugins/deep-think/ by default.
# - --symlink creates a symlink instead (dev mode: edits to this repo are live).
# - Respects $HERMES_HOME if set; defaults to ~/.hermes.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_SRC="$REPO_DIR/plugin"

HERMES_HOME_ARG=""
USE_SYMLINK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hermes-home) HERMES_HOME_ARG="$2"; shift 2 ;;
    --symlink) USE_SYMLINK=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

HERMES_HOME="${HERMES_HOME_ARG:-${HERMES_HOME:-$HOME/.hermes}}"
DEST="$HERMES_HOME/plugins/deep-think"

if [[ ! -f "$PLUGIN_SRC/plugin.yaml" ]]; then
  echo "error: $PLUGIN_SRC/plugin.yaml not found — run from the repo root" >&2
  exit 1
fi

mkdir -p "$HERMES_HOME/plugins"

if [[ -e "$DEST" ]] && [[ ! -L "$DEST" ]]; then
  echo ">> existing install found at $DEST — replacing"
  rm -rf "$DEST"
fi

if [[ "$USE_SYMLINK" -eq 1 ]]; then
  ln -sfn "$PLUGIN_SRC" "$DEST"
  echo ">> symlinked $DEST → $PLUGIN_SRC (dev mode)"
else
  cp -r "$PLUGIN_SRC" "$DEST"
  echo ">> copied plugin to $DEST"
fi

echo
echo "Next steps:"
echo "  1. hermes plugins enable deep-think"
echo "  2. hermes plugins list | grep deep-think   # verify"
echo "  3. start a new session (or /reset) — the tool appears in new sessions"
echo "  4. tail -f $HERMES_HOME/plugin-data/deep-think/reasoning-trace.jsonl"
