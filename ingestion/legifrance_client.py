"""
legifrance_client.py  —  Retrieves all articles of the Civil Code via the PISTE API.
"""

import os, time, json, requests
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ── Sandbox or production environment ────────────────────────
# Set LEGIFRANCE_ENV=production in .env once you have
# the production keys. Default: sandbox.
_ENV = os.getenv("LEGIFRANCE_ENV", "sandbox").lower()

if _ENV == "production":
    TOKEN_URL = "https://oauth.piste.gouv.fr/api/oauth/token"
    API_BASE  = "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app"
else:
    TOKEN_URL = "https://sandbox-oauth.piste.gouv.fr/api/oauth/token"
    API_BASE  = "https://sandbox-api.piste.gouv.fr/dila/legifrance/lf-engine-app"

print(f"🌍 Légifrance environment: {_ENV.upper()}")

CODE_CIVIL_ID  = "LEGITEXT000006070721"
CACHE_FILE     = Path(__file__).parent / "cache_articles.json"


class LegifranceClient:

    def __init__(self):
        self.client_id     = os.getenv("LEGIFRANCE_CLIENT_ID")
        self.client_secret = os.getenv("LEGIFRANCE_CLIENT_SECRET")
        self.token         = None
        self.token_expiry  = 0
        missing = [
            name for name, val in (
                ("LEGIFRANCE_CLIENT_ID",     self.client_id),
                ("LEGIFRANCE_CLIENT_SECRET", self.client_secret),
            )
            if not val or "your_" in val.lower()
        ]
        if missing:
            raise ValueError(
                "❌ Missing or placeholder Légifrance credentials in .env: "
                + ", ".join(missing)
                + "\n   → configuration template: env.sample"
                "\n     cp env.sample .env then fill in the values"
                "\n   → keys to create at https://piste.gouv.fr"
            )

    # ── Auth ─────────────────────────────────────────────────

    def _get_token(self) -> str:
        if self.token and time.time() < self.token_expiry - 60:
            return self.token
        print("🔑 Légifrance authentication...")
        r = requests.post(TOKEN_URL, data={
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "openid",
        })
        r.raise_for_status()
        d = r.json()
        self.token        = d["access_token"]
        self.token_expiry = time.time() + d.get("expires_in", 3600)
        return self.token

    def _h(self): return {"Authorization": f"Bearer {self._get_token()}", "Content-Type": "application/json"}

    # ── Structure & articles ─────────────────────────────────

    def get_structure(self) -> dict:
        r = requests.post(f"{API_BASE}/consult/code/tableMatieres",
            json={"textId": CODE_CIVIL_ID, "date": int(time.time() * 1000)},
            headers=self._h())
        r.raise_for_status()
        return r.json()

    def get_article(self, article_id: str) -> dict | None:
        try:
            r = requests.post(f"{API_BASE}/consult/getArticle",
                json={"id": article_id}, headers=self._h())
            r.raise_for_status()
            a = r.json().get("article", {})
            if a.get("etat", "") in ("ABROGE", "ABROGE_DIFF", "REMPLACE"):
                return None
            return a
        except Exception:
            return None

    def _collect_refs(self, node: dict, crumb: list) -> list[dict]:
        refs, typ, title = [], node.get("type",""), node.get("titre", node.get("intitule",""))
        cur = crumb + ([title] if typ != "ARTICLE" and title else [])
        if typ == "ARTICLE":
            refs.append({"id": node.get("id"), "num": node.get("num",""), "breadcrumb": " > ".join(crumb)})
        for child in node.get("sections",[]) + node.get("articles",[]):
            refs.extend(self._collect_refs(child, cur))
        return refs

    # ── Pipeline ─────────────────────────────────────────────

    def fetch_all(self, use_cache=False) -> list[dict]:
        if use_cache and CACHE_FILE.exists():
            print(f"📂 Cache found: {CACHE_FILE}")
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))

        refs = self._collect_refs(self.get_structure(), [])
        print(f"📋 {len(refs)} articles in the table of contents")

        articles, skipped = [], 0
        for i, ref in enumerate(refs, 1):
            if i % 100 == 0: print(f"  ⏳ {i}/{len(refs)}...")
            raw = self.get_article(ref["id"])
            if raw is None: skipped += 1; continue

            html = raw.get("texteHtml", raw.get("texte",""))
            texte = BeautifulSoup(html, "lxml").get_text(" ").strip() if html else ""
            if not texte: skipped += 1; continue

            articles.append({
                "id":         raw.get("id",""),
                "num":        raw.get("num", ref["num"]),
                "titre":      raw.get("titre",""),
                "texte":      texte,
                "breadcrumb": ref["breadcrumb"],
                "etat":       raw.get("etat","VIGUEUR"),
            })
            time.sleep(0.2)

        print(f"✅ {len(articles)} articles retrieved ({skipped} skipped)")
        CACHE_FILE.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
        return articles
