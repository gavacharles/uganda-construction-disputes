#!/usr/bin/env bash
# One-way backup of results and private notes to OneDrive (never syncs back).
# Usage: scripts/backup.sh          -> outputs/ + notes
#        scripts/backup.sh --data   -> also a dated zip of data/interim + data/processed
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="$HOME/Library/CloudStorage/OneDrive-Personal(2)/PhD/UJ/Research/Papers/Temporal Analysis/backup"
mkdir -p "$DEST"
rsync -a --exclude '.DS_Store' outputs/ "$DEST/outputs/"
for f in CLAUDE.md COMPENDIUM_PLAN.md codebook.md; do [ -f "$f" ] && cp -p "$f" "$DEST/"; done
if [ "${1:-}" = "--data" ]; then
  mkdir -p "$DEST/data_snapshots"
  zip -qr "$DEST/data_snapshots/data_$(date +%Y%m%d).zip" data/interim data/processed -x '*.DS_Store'
fi
date "+%Y-%m-%d %H:%M" > "$DEST/LAST_BACKUP.txt"
echo "Backed up to: $DEST"
