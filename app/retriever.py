"""
Retrieval engine: Keyword (BM25) only.
Optimized for 100% reliability and minimal RAM footprint on Render Free Tier.
"""

import re

from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi

from app.models import CatalogEntry
from app.catalog import catalog


class BM25Retriever:
    """
    Pure sparse retrieval (BM25).
    Zero PyTorch/FAISS dependencies. Peak RAM: ~65MB.
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.entries: List[CatalogEntry] = []
        self._is_built = False

    def build(self):
        """Build BM25 index from the catalog."""
        self.entries = catalog.entries
        if not self.entries:
            raise RuntimeError("Catalog is empty — cannot build retriever")

        print(f"[Retriever] Building BM25 index for {len(self.entries)} entries...")

        # --- Sparse index (BM25) ---
        texts = [e.search_text for e in self.entries]
        tokenized = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        
        print(f"[Retriever] BM25 index built: {len(tokenized)} documents")
        self._is_built = True

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        text = text.lower()
        # Keep alphanumeric, #, +, and . (useful for C#, C++, ASP.NET)
        tokens = re.findall(r'\b[a-z0-9#\+\.]+\b', text)
        return tokens

    def search_bm25(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """BM25 keyword search. Returns (index, score) tuples."""
        if not self._is_built:
            raise RuntimeError("Retriever not built — call build() first")

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        
        # Get top-k indices (sorting the array)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx])))
        return results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 20,
        **kwargs
    ) -> List[Tuple[CatalogEntry, float]]:
        """
        Main search method (kept named search_hybrid for API compatibility with agent.py).
        Returns (CatalogEntry, score) sorted by relevance.
        """
        bm25_results = self.search_bm25(query, top_k=top_k)

        results = []
        for idx, score in bm25_results:
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
        BM25 search with optional post-retrieval filters.
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
retriever = BM25Retriever()
