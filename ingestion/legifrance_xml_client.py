"""
legifrance_xml_client.py
────────────────────────
Parses the Civil Code from the official Légifrance XML files
(downloaded from https://echanges.dila.gouv.fr/LEGI/)

Replaces legifrance_client.py — same interface, no API needed.

Download:
  wget https://echanges.dila.gouv.fr/LEGI/legi.tar.gz
  # or look for the latest dump at:
  # https://echanges.dila.gouv.fr/LEGI/

Usage:
  from ingestion.legifrance_xml_client import LegifranceClient
  client = LegifranceClient()
  articles = client.fetch_all()
"""

import os, json, tarfile, zipfile
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

# Official ID of the Civil Code in the LEGI files
CODE_CIVIL_ID  = "LEGITEXT000006070721"
CACHE_FILE     = Path(__file__).parent / "cache_articles.json"

# Look for the XML files in these folders (in order of priority)
SEARCH_PATHS = [
    Path("./legi-data"),           # git clone legifrance/legi-data
    Path("./legi"),                # folder extracted from the tar.gz
    Path("./data"),
    Path("."),
]


class LegifranceClient:

    def __init__(self):
        self.xml_root = self._find_xml_root()
        if self.xml_root:
            print(f"📁 XML files found: {self.xml_root}")
        else:
            print("⚠️  No XML file found locally")
            print("   Options:")
            print("   1. wget https://echanges.dila.gouv.fr/LEGI/legi.tar.gz && tar xzf legi.tar.gz")
            print("   2. git clone https://github.com/legifrance/legi-data")

    def _find_xml_root(self) -> Path | None:
        """Looks for the folder containing the LEGI XML files."""
        for base in SEARCH_PATHS:
            # Look for a typical Civil Code XML file
            matches = list(base.rglob(f"*{CODE_CIVIL_ID}*.xml"))
            if matches:
                return matches[0].parent
            # Look for the legi/ folder
            legi_dir = base / "legi" / "global" / "code_et_TNC_en_vigueur"
            if legi_dir.exists():
                return legi_dir
        return None

    def _parse_article_xml(self, xml_path: Path) -> dict | None:
        """Parses a Légifrance article XML file."""
        try:
            with open(xml_path, encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "xml")

            # Check that the article is in force
            etat = soup.find("ETAT")
            if etat and etat.text in ("ABROGE", "ABROGE_DIFF", "REMPLACE"):
                return None

            # Extract the article number
            num_tag = soup.find("NUM")
            num = num_tag.text.strip() if num_tag else ""

            # Extract the title
            titre_tag = soup.find("TITRE")
            titre = titre_tag.text.strip() if titre_tag else ""

            # Extract the text (may be in HTML)
            texte_tag = soup.find("BLOC_TEXTUEL") or soup.find("TEXTE")
            if not texte_tag:
                return None

            html_content = str(texte_tag)
            texte = BeautifulSoup(html_content, "lxml").get_text(" ").strip()

            if not texte or len(texte) < 10:
                return None

            # Extract the ID
            article_id = soup.find("ID")
            art_id = article_id.text.strip() if article_id else xml_path.stem

            return {
                "id":         art_id,
                "num":        num,
                "titre":      titre,
                "texte":      texte,
                "breadcrumb": self._get_breadcrumb(xml_path),
                "etat":       "VIGUEUR",
            }
        except Exception as e:
            return None

    def _get_breadcrumb(self, xml_path: Path) -> str:
        """Builds the breadcrumb from the file path."""
        parts = xml_path.parts
        # LEGI folders follow the structure:
        # .../code_civil/LEGI/SCTA/SCTA/SCTA/LEGIARTI...xml
        # We extract the readable parts
        breadcrumb_parts = []
        for part in parts:
            if part.startswith("LEGI") and len(part) > 10:
                continue
            if part in ("global", "code_et_TNC_en_vigueur", "legi"):
                continue
            if part.endswith(".xml"):
                continue
            if len(part) > 3 and not part.startswith("SCTA"):
                breadcrumb_parts.append(part)
        return " > ".join(breadcrumb_parts[-3:]) if breadcrumb_parts else ""

    def fetch_all(self, use_cache=False) -> list[dict]:
        """
        Main entry point — same interface as the API client.
        """
        if use_cache and CACHE_FILE.exists():
            print(f"📂 Cache found: {CACHE_FILE}")
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))

        if not self.xml_root:
            raise FileNotFoundError(
                "❌ No Légifrance XML file found.\n"
                "   Download the data:\n"
                "   wget https://echanges.dila.gouv.fr/LEGI/legi.tar.gz\n"
                "   tar xzf legi.tar.gz"
            )

        # Find all Civil Code article XML files
        print(f"🔍 Searching for XML articles in {self.xml_root}...")
        xml_files = []

        # Strategy 1: search via the Civil Code ID
        code_dirs = list(Path(".").rglob(f"*{CODE_CIVIL_ID}*"))
        for d in code_dirs:
            if d.is_dir():
                xml_files.extend(d.rglob("LEGIARTI*.xml"))

        # Strategy 2: search in the folder found
        if not xml_files and self.xml_root:
            xml_files = list(self.xml_root.rglob("LEGIARTI*.xml"))

        print(f"📋 {len(xml_files)} XML files found")

        articles, skipped = [], 0
        for xml_path in tqdm(xml_files, desc="Parsing XML"):
            article = self._parse_article_xml(xml_path)
            if article:
                articles.append(article)
            else:
                skipped += 1

        print(f"✅ {len(articles)} articles parsed ({skipped} skipped/repealed)")

        if articles:
            CACHE_FILE.write_text(
                json.dumps(articles, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"💾 Cache saved: {CACHE_FILE}")

        return articles


# ── Alternative: GitHub data ────────────────────────────────

def download_from_github():
    """
    Clones the legi-data GitHub repository which contains the Civil Code in XML.
    Simpler than the official tar.gz.
    """
    import subprocess
    print("📥 Cloning the legi-data repository (may take a few minutes)...")
    result = subprocess.run([
        "git", "clone", "--depth=1",
        "https://github.com/legifrance/legi-data",
        "./legi-data"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ legi-data cloned successfully")
    else:
        print(f"❌ Error: {result.stderr}")
        print("   Try manually: git clone https://github.com/legifrance/legi-data")


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        download_from_github()
    else:
        client = LegifranceClient()
        articles = client.fetch_all()
        print(f"\n📊 {len(articles)} articles available")
        if articles:
            print("\nExamples:")
            for a in articles[:3]:
                print(f"  Art. {a['num']} — {a['texte'][:80]}...")
