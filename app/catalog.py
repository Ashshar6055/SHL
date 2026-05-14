"""
Catalog loading and lookup utilities.
Loads the SHL product catalog from JSON and provides lookup functions.
"""

import json
import os
from typing import Dict, List, Optional
from app.models import CatalogEntry


class Catalog:
    """Singleton-style catalog manager."""

    def __init__(self):
        self._entries: List[CatalogEntry] = []
        self._by_name: Dict[str, CatalogEntry] = {}
        self._by_id: Dict[str, CatalogEntry] = {}
        self._by_url: Dict[str, CatalogEntry] = {}

    def load(self, path: str = None):
        """Load catalog from JSON file."""
        if path is None:
            # Try multiple paths
            candidates = [
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "shlcatalog.json"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "shlcatalog.json"),
                "shlcatalog.json",
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    path = candidate
                    break
            if path is None:
                raise FileNotFoundError(f"Catalog not found. Tried: {candidates}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self._entries = []
        self._by_name = {}
        self._by_id = {}
        self._by_url = {}

        for item in raw_data:
            entry = CatalogEntry(
                entity_id=str(item.get("entity_id", "")),
                name=item.get("name", ""),
                link=item.get("link", ""),
                job_levels=item.get("job_levels", []),
                languages=item.get("languages", []),
                duration=item.get("duration", ""),
                remote=item.get("remote", ""),
                adaptive=item.get("adaptive", ""),
                description=item.get("description", ""),
                keys=item.get("keys", []),
            )
            self._entries.append(entry)
            self._by_name[entry.name.lower()] = entry
            self._by_id[entry.entity_id] = entry
            self._by_url[entry.link.lower()] = entry

        print(f"[Catalog] Loaded {len(self._entries)} assessments")

    @property
    def entries(self) -> List[CatalogEntry]:
        return self._entries

    def get_by_name(self, name: str) -> Optional[CatalogEntry]:
        """Exact name lookup (case-insensitive)."""
        return self._by_name.get(name.lower())

    def get_by_id(self, entity_id: str) -> Optional[CatalogEntry]:
        """Lookup by entity ID."""
        return self._by_id.get(str(entity_id))

    def get_by_url(self, url: str) -> Optional[CatalogEntry]:
        """Lookup by URL."""
        return self._by_url.get(url.lower())

    def search_by_name(self, query: str) -> List[CatalogEntry]:
        """Fuzzy name search — returns entries whose names contain the query."""
        query_lower = query.lower()
        return [e for e in self._entries if query_lower in e.name.lower()]

    def filter_by_test_type(self, test_type_key: str) -> List[CatalogEntry]:
        """Filter by test type key (e.g., 'Knowledge & Skills')."""
        return [e for e in self._entries if test_type_key in e.keys]

    def filter_by_job_level(self, level: str) -> List[CatalogEntry]:
        """Filter by job level."""
        level_lower = level.lower()
        return [e for e in self._entries if any(level_lower in jl.lower() for jl in e.job_levels)]

    def validate_recommendation(self, name: str, url: str) -> Optional[CatalogEntry]:
        """
        Validate that a recommendation exists in the catalog.
        Returns the matching entry or None.
        """
        # Try exact URL match first
        entry = self.get_by_url(url)
        if entry:
            return entry

        # Try exact name match
        entry = self.get_by_name(name)
        if entry:
            return entry

        # Try fuzzy name match
        matches = self.search_by_name(name)
        if len(matches) == 1:
            return matches[0]

        return None


# Global catalog instance
catalog = Catalog()
