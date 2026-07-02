"""
Phase 2 — In-memory LRU cache for parsed queries.

Avoids redundant Claude API calls for identical or normalised queries.
The cache key is the lowercased, whitespace-collapsed query string.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Optional

from src.core.schema import ParsedQuery # pyrefly: ignore [missing-import]


def _normalise(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


class QueryCache:
    """Thread-safe LRU cache keyed on normalised query strings."""

    def __init__(self, max_size: int = 256):
        self._max_size = max_size
        self._store: OrderedDict[str, ParsedQuery] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, query: str) -> Optional[ParsedQuery]:
        key = _normalise(query)
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def put(self, query: str, parsed: ParsedQuery) -> None:
        key = _normalise(query)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = parsed
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
