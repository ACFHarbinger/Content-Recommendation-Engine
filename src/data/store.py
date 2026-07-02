"""
SQLite-backed vector store — no external database required.

SQLiteStore manages a single ``items`` table in a local .db file
(default ``data/rec_engine.db``) or an in-memory database (``":memory:"``).
Dense and sparse vectors are serialised as JSON TEXT blobs; metadata fields
are stored as native SQLite types with JSON arrays for list columns.

Public API mirrors the retired QdrantStore so the rest of the pipeline
requires minimal changes:
  create_collection()  — CREATE TABLE IF NOT EXISTS  (idempotent)
  upsert()             — INSERT OR REPLACE batches
  delete()             — DELETE WHERE id = ?
  collection_info()    — row count + db path
  fetch_all()          — SELECT * with optional WHERE (includes vectors)
  fetch_filtered()     — SELECT metadata columns only (for scroll / history)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from src.core.config import Settings, get_settings
from src.core.schema import EmbeddedItem

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS items (
    id                    TEXT PRIMARY KEY,
    title                 TEXT NOT NULL,
    type                  TEXT,
    watch_status          TEXT,
    rating                REAL,
    year_released         INTEGER,
    num_episodes_or_pages INTEGER,
    genres                TEXT,               -- JSON array
    tags                  TEXT,               -- JSON array
    associated_entities   TEXT,               -- JSON array
    local_file_location   TEXT,
    web_link              TEXT,
    dense_vector          TEXT NOT NULL,      -- JSON array[float]
    sparse_indices        TEXT NOT NULL,      -- JSON array[int]
    sparse_values         TEXT NOT NULL       -- JSON array[float]
)
"""

_JSON_COLS = (
    "genres",
    "tags",
    "associated_entities",
    "dense_vector",
    "sparse_indices",
    "sparse_values",
)
_META_JSON_COLS = ("genres", "tags", "associated_entities")


class SQLiteStore:
    """
    Manages the SQLite items table for the recommendation engine.

    Usage
    -----
    store = SQLiteStore()
    store.create_collection()          # idempotent
    store.upsert(embedded_items)
    info = store.collection_info()
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._cfg = settings or get_settings()
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            db_path = self._cfg.sqlite_path
            if db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(db_path, check_same_thread=False)
                self._conn.execute("PRAGMA journal_mode=WAL")
            else:
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            logger.info("Connected to SQLite at %s", db_path)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self) -> bool:
        """
        Create the items table if it does not exist.  Idempotent.

        Returns True when the table was created, False when it already existed.
        """
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='items'"
        )
        existed = cur.fetchone() is not None
        conn.execute(_CREATE_TABLE)
        conn.commit()
        if not existed:
            logger.info("Created 'items' table at %s", self._cfg.sqlite_path)
        return not existed

    def collection_info(self) -> dict:
        """Return row count and database path."""
        conn = self._get_conn()
        (count,) = conn.execute("SELECT COUNT(*) FROM items").fetchone()
        return {
            "points_count": count,
            "collection": "items",
            "storage": self._cfg.sqlite_path,
        }

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(self, items: list[EmbeddedItem], batch_size: int = 64) -> int:
        """
        Upsert a list of EmbeddedItems to SQLite.

        Returns the number of rows inserted or replaced.
        """
        conn = self._get_conn()
        upserted = 0

        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            rows = []
            for ei in batch:
                item = ei.item
                rows.append(
                    (
                        item.id,
                        item.title,
                        item.type or "",
                        item.watch_status or "",
                        item.rating,
                        item.year_released,
                        item.num_episodes_or_pages,
                        json.dumps(item.genres),
                        json.dumps(item.tags),
                        json.dumps(item.associated_entities),
                        item.local_file_location or "",
                        item.web_link or "",
                        json.dumps(ei.dense_vector),
                        json.dumps(ei.sparse_indices),
                        json.dumps(ei.sparse_values),
                    )
                )

            conn.executemany(
                """
                INSERT OR REPLACE INTO items (
                    id, title, type, watch_status, rating, year_released,
                    num_episodes_or_pages, genres, tags, associated_entities,
                    local_file_location, web_link,
                    dense_vector, sparse_indices, sparse_values
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            upserted += len(rows)
            logger.debug("Upserted %d/%d rows", upserted, len(items))

        return upserted

    def delete(self, item_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        conn.commit()

    # ------------------------------------------------------------------
    # Read helpers used by retriever and pipeline
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        where_clause: str = "",
        params: Optional[list] = None,
    ) -> list[dict]:
        """
        Fetch all rows (including vectors) matching the optional WHERE clause.

        JSON columns are decoded back to Python lists before returning.
        """
        conn = self._get_conn()
        sql = "SELECT * FROM items"
        if where_clause:
            sql += f" WHERE {where_clause}"
        cur = conn.execute(sql, params or [])
        rows = []
        for row in cur.fetchall():
            d = dict(row)
            for col in _JSON_COLS:
                raw = d.get(col)
                d[col] = json.loads(raw) if raw else []
            rows.append(d)
        return rows

    def fetch_filtered(
        self,
        where_clause: str = "",
        params: Optional[list] = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        Fetch metadata rows (without vectors) for scroll and history profile.

        JSON columns are decoded back to Python lists before returning.
        """
        conn = self._get_conn()
        sql = (
            "SELECT id, title, type, watch_status, rating, year_released, "
            "num_episodes_or_pages, genres, tags, associated_entities, "
            "local_file_location, web_link FROM items"
        )
        if where_clause:
            sql += f" WHERE {where_clause}"
        sql += f" LIMIT {limit}"
        cur = conn.execute(sql, params or [])
        rows = []
        for row in cur.fetchall():
            d = dict(row)
            for col in _META_JSON_COLS:
                raw = d.get(col)
                d[col] = json.loads(raw) if raw else []
            rows.append(d)
        return rows
