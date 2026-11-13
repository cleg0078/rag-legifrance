#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  fetch_legi_dump.sh — download and extract the Légifrance dump (legi/)
#
#  Used by `make legi-dump` and `make articles-cache` (source=dump).
#  No data is overwritten: if legi/ already contains LEGIARTI*.xml files,
#  the script exits immediately.
#
#  Usage: fetch_legi_dump.sh [URL_OPENDATA] [DESTINATION]
# ══════════════════════════════════════════════════════════════
set -euo pipefail

URL="${1:-https://echanges.dila.gouv.fr/OPENDATA/LEGI/}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${2:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [ -n "$(find "$DEST/legi" -name 'LEGIARTI*.xml' -print -quit 2>/dev/null)" ]; then
    echo "✅ LEGI dump already extracted ($DEST/legi)"
    exit 0
fi

echo "ℹ  Searching for the latest LEGI archive..."
LATEST=$(curl -s "$URL" | grep -oE 'Freemium_legi_global_[0-9]{8}-[0-9]{6}\.tar\.gz' | sort -u | tail -1)
if [ -z "$LATEST" ]; then
    echo "❌ No Freemium_legi_global_*.tar.gz archive at $URL" >&2
    exit 1
fi

ARCHIVE="$DEST/$LATEST"
echo "ℹ  Downloading $URL$LATEST (~1.2 GB)..."
curl -L --fail --retry 3 -C - -o "$ARCHIVE" "$URL$LATEST"

echo "ℹ  Extracting $LATEST (~27 GB)..."
tar xzf "$ARCHIVE" -C "$DEST"

if [ -z "$(find "$DEST/legi" -name 'LEGIARTI*.xml' -print -quit 2>/dev/null)" ]; then
    echo "❌ Incomplete extraction: no LEGIARTI*.xml under $DEST/legi" >&2
    exit 1
fi
echo "✅ Légifrance dump downloaded and extracted ($DEST/legi)"
