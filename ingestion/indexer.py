"""
indexer.py  —  Cleaning, chunking, embeddings → Qdrant.
"""

import os, re, uuid
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

QDRANT_HOST    = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION     = os.getenv("QDRANT_COLLECTION", "code_civil")
EMB_MODEL      = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
EMB_DIM        = int(os.getenv("EMBEDDING_DIMENSION", 1024))
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 50))
BATCH          = 32


# ── Cleaning ─────────────────────────────────────────────────

def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    for src, dst in [("\u2019","'"),("\u2018","'"),("\u00ab",'"'),("\u00bb",'"'),
                     ("\u2013","-"),("\u2014","-")]:
        text = text.replace(src, dst)
    return re.sub(r"[\x00-\x1f\x7f]", "", text).strip()


# ── Chunking ─────────────────────────────────────────────────

def chunk_article(article: dict) -> list[dict]:
    texte = clean(article.get("texte",""))
    if len(texte) < 20:
        return []

    num, titre = article.get("num",""), article.get("titre","")
    header     = f"Article {num}" + (f" — {titre}" if titre else "")
    full_text  = f"{header}\n{texte}"
    words      = full_text.split()

    if len(words) <= CHUNK_SIZE:
        parts = [full_text]
    else:
        parts, start = [], 0
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            parts.append(" ".join(words[start:end]))
            if end == len(words): break
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return [{
        "text": p,
        "metadata": {
            "article_id":  article.get("id",""),
            "num":         num,
            "titre":       titre,
            "breadcrumb":  article.get("breadcrumb",""),
            "chunk_index": i,
            "chunk_total": len(parts),
            "source":      "French Civil Code — Légifrance",
        }
    } for i, p in enumerate(parts)]


# ── Embedder ─────────────────────────────────────────────────

class Embedder:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔄 Loading {EMB_MODEL} on {device}...")
        self.model = SentenceTransformer(EMB_MODEL, device=device)
        print(f"✅ Embedder ready (dim={self.model.get_sentence_embedding_dimension()})")

    @property
    def dim(self): return self.model.get_sentence_embedding_dimension()

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        all_emb  = []
        for i in tqdm(range(0, len(prefixed), BATCH), desc="Embeddings"):
            batch = prefixed[i:i+BATCH]
            emb   = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_emb.extend(emb.tolist())
        return all_emb

    def encode_query(self, query: str) -> list[float]:
        return self.model.encode(f"query: {query}", normalize_embeddings=True).tolist()


# ── Qdrant ───────────────────────────────────────────────────

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def create(self, dim: int, recreate=False):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION in existing:
            if recreate:
                self.client.delete_collection(COLLECTION)
            else:
                print(f"ℹ️  Collection '{COLLECTION}' already exists — skipping")
                return
        self.client.create_collection(COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        for field in ("num", "breadcrumb"):
            self.client.create_payload_index(COLLECTION, field, "keyword")
        print(f"✅ Collection '{COLLECTION}' created")

    def upsert(self, chunks_with_emb: list[dict]):
        points = [PointStruct(
            id=str(uuid.uuid4()),
            vector=c["embedding"],
            payload={"text": c["text"], **c["metadata"]}
        ) for c in chunks_with_emb]
        for i in tqdm(range(0, len(points), 100), desc="Qdrant indexing"):
            self.client.upsert(COLLECTION, points=points[i:i+100])
        print(f"✅ {self.client.count(COLLECTION).count} points in Qdrant")

    def search(self, vector: list[float], top_k=5) -> list[dict]:
        # qdrant-client >= 1.16: QdrantClient.search() was removed.
        # Replaced by query_points() → QueryResponse.points (points = ScoredPoint).
        results = self.client.query_points(COLLECTION, query=vector,
                                           limit=top_k, with_payload=True).points
        return [{"score": r.score, **(r.payload or {})} for r in results]


# ── Full pipeline ────────────────────────────────────────────

def run_ingestion(articles: list[dict], recreate=False):
    # Chunking
    print("\n✂️  Chunking...")
    chunks = []
    for a in articles:
        chunks.extend(chunk_article(a))
    print(f"   → {len(chunks)} chunks generated")

    # Embeddings
    embedder = Embedder()
    texts    = [c["text"] for c in chunks]
    embs     = embedder.encode_passages(texts)
    for c, e in zip(chunks, embs):
        c["embedding"] = e

    # Indexing
    store = VectorStore()
    store.create(dim=embedder.dim, recreate=recreate)
    store.upsert(chunks)

    return embedder, store
