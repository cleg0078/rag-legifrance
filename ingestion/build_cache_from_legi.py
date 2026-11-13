#!/usr/bin/env python3
"""build_cache_from_legi.py — rebuilds ingestion/cache_articles.json from the extracted LEGI dump.

Used by `make articles-cache` (source=dump) when the PISTE credentials are
not available: no network request, no API call.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.legifrance_xml_client import CACHE_FILE, CODE_CIVIL_ID, LegifranceClient


def find_code_civil_dir():
    base = ROOT / "legi"
    if not base.is_dir():
        return None
    for path in base.rglob(f"*{CODE_CIVIL_ID}*"):
        if path.is_dir():
            return path
    return None


def main() -> int:
    root = find_code_civil_dir()
    if root is None:
        print(f"❌ LEGI dump not found under legi/ (folder *{CODE_CIVIL_ID}*)", file=sys.stderr)
        print("   → make legi-dump", file=sys.stderr)
        return 1

    files = sorted(root.rglob("LEGIARTI*.xml"))
    print(f"📄 {len(files)} XML files (Civil Code)")
    if not files:
        print("❌ No LEGIARTI*.xml file found — incomplete dump?", file=sys.stderr)
        return 1

    # __new__ to avoid __init__ (which would do a costly rglob over legi/).
    client = LegifranceClient.__new__(LegifranceClient)
    articles = [a for f in files if (a := client._parse_article_xml(f))]
    print(f"✅ {len(articles)} articles in force")
    if not articles:
        print("❌ No article parsed — incomplete dump?", file=sys.stderr)
        return 1

    CACHE_FILE.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Cache: {CACHE_FILE} ({CACHE_FILE.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
