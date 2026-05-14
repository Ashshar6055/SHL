"""
Hybrid retrieval engine: Semantic (FAISS + sentence-transformers) + Keyword (BM25).
Results fused using Reciprocal Rank Fusion for robust recall.
"""

import numpy as np
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

from app.models import CatalogEntry
from app.catalog import catalog


class HybridRetriever:
    """
    Combines dense retrieval (FAISS) with sparse retrieval (BM25)
    using Reciprocal Rank Fusion for better recall.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.bm25: Optional[BM25Okapi] = None
        self.entries: List[CatalogEntry] = []
        self._is_built = False

    def build(self):
        """Build both FAISS and BM25 indices from the catalog."""
        self.entries = catalog.entries
        if not self.entries:
            raise RuntimeError("Catalog is empty — cannot build retriever")

        print(f"[Retriever] Building indices for {len(self.entries)} entries...")

        # --- Dense index (FAISS) ---
        print(f"[Retriever] Loading sentence-transformer: {self.model_name}")
        self.encoder = SentenceTransformer(self.model_name)

        texts = [e.search_text for e in self.entries]
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=64,
        )
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized)
        self.faiss_index.add(embeddings)
        print(f"[Retriever] FAISS index built: {self.faiss_index.ntotal} vectors, dim={dim}")

        # --- Sparse index (BM25) ---
        tokenized = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[Retriever] BM25 index built: {len(tokenized)} documents")

        self._is_built = True

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase, split on whitespace and common delimiters
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9#\+\.]+\b', text)
        return tokens

    def search_semantic(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Semantic search using FAISS. Returns (index, score) tuples."""
        if not self._is_built:
            raise RuntimeError("Retriever not built — call build() first")

        query_vec = self.encoder.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.faiss_index.search(query_vec, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                results.append((int(idx), float(score)))
        return results

    def search_bm25(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """BM25 keyword search. Returns (index, score) tuples."""
        if not self._is_built:
            raise RuntimeError("Retriever not built — call build() first")

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx])))
        return results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 15,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
    ) -> List[Tuple[CatalogEntry, float]]:
        """
        Hybrid search using Reciprocal Rank Fusion.
        Returns (CatalogEntry, fused_score) sorted by relevance.
        """
        # Fetch more candidates than needed for fusion
        fetch_k = max(top_k * 3, 30)

        semantic_results = self.search_semantic(query, top_k=fetch_k)
        bm25_results = self.search_bm25(query, top_k=fetch_k)

        # Reciprocal Rank Fusion
        rrf_scores = {}

        for rank, (idx, _) in enumerate(semantic_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + semantic_weight / (rrf_k + rank + 1)

        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + bm25_weight / (rrf_k + rank + 1)

        # Sort by fused score
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in sorted_indices[:top_k]:
            if idx < len(self.entries):
                results.append((self.entries[idx], score))

        return results

    def search_with_filters(
        self,
        query: str,
        top_k: int = 15,
        job_level: Optional[str] = None,
        test_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[Tuple[CatalogEntry, float]]:
        """
        Hybrid search with optional post-retrieval filters.
        Filters are applied AFTER retrieval to avoid missing relevant results.
        """
        # Retrieve more to compensate for filtering
        raw_results = self.search_hybrid(query, top_k=top_k * 3)

        if not any([job_level, test_type, language]):
            return raw_results[:top_k]

        filtered = []
        for entry, score in raw_results:
            if job_level and not any(job_level.lower() in jl.lower() for jl in entry.job_levels):
                continue
            if test_type and test_type not in entry.keys:
                continue
            if language and not any(language.lower() in lang.lower() for lang in entry.languages):
                continue
            filtered.append((entry, score))

        return filtered[:top_k]


# Global retriever instance
retriever = HybridRetriever()
