"""Multilingual embedding search over KB chunks, with document-precedence adjustments."""
import hashlib
import os
import numpy as np
from .config import ROOT
from .kb import load_chunks

MODEL_NAME = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
INDEX_DIR = ROOT / ".index"
TYPE_BONUS = {"policy": 0.04, "agent_guide": 0.03, "help": 0.01, "faq": 0.0, "marketing": -0.03, "meta": -0.02}
SUPERSEDED_PENALTY = 0.04


def _e5():
    return "e5" in MODEL_NAME.lower()


class Index:
    def __init__(self):
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL_NAME)
        self.chunks = load_chunks()
        texts = [("passage: " if _e5() else "") + c["embed_text"] for c in self.chunks]
        key = hashlib.sha256((MODEL_NAME + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
        INDEX_DIR.mkdir(exist_ok=True)
        path = INDEX_DIR / f"emb_{key}.npy"
        if path.exists():
            self.emb = np.load(path)
        else:
            self.emb = self.model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
            np.save(path, self.emb)
        self.adj = np.array([
            TYPE_BONUS.get(c["doc_type"], 0.0) - (SUPERSEDED_PENALTY if c["status"] != "current" else 0.0)
            for c in self.chunks])

    def _encode(self, queries):
        qs = [q for q in queries if q and q.strip()]
        return self.model.encode([("query: " if _e5() else "") + q for q in qs], normalize_embeddings=True)

    def search(self, queries, k=6):
        """Best score across queries (max fusion)."""
        sims = (self._encode(queries) @ self.emb.T).max(axis=0)
        order = np.argsort(-(sims + self.adj))[:k]
        return [self.chunks[i] for i in order]

    def search_rrf(self, queries, k=8, rrf_k=30):
        """Rank each query separately, then merge by reciprocal rank so one query cannot push out another's hits."""
        sims = self._encode(queries) @ self.emb.T
        scores = {}
        for row in sims:
            order = np.argsort(-(row + self.adj))[:max(2 * k, 20)]
            for rank, i in enumerate(order):
                scores[i] = scores.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        top = sorted(scores, key=lambda i: -scores[i])[:k]
        return [self.chunks[i] for i in top]


_INDEX = None


def get_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = Index()
    return _INDEX
