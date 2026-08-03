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

from src.core.config import Settings, get_settings  # pyrefly: ignore [missing-import]
from src.core.schema import EmbeddedItem  # pyrefly: ignore [missing-import]

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


_ALL_COLS = (
    "id", "title", "type", "watch_status", "rating", "year_released",
    "num_episodes_or_pages", "genres", "tags", "associated_entities",
    "local_file_location", "web_link", "dense_vector", "sparse_indices",
    "sparse_values",
)
_META_COLS = (
    "id", "title", "type", "watch_status", "rating", "year_released",
    "num_episodes_or_pages", "genres", "tags", "associated_entities",
    "local_file_location", "web_link",
)


class EncryptedSQLiteStore:
    """
    Drop-in replacement for :class:`SQLiteStore`, backed by an already-open
    ``base.database.Database`` handle (Image Toolkit's unified, encrypted
    SQLCipher store) instead of a plaintext ``sqlite3`` file.

    Storage-layer swap only -- the SQL text and the calling code's
    retrieval/scoring logic (HybridRetriever, Scorer, RRF fusion) are
    completely unaware of the difference; only ``upsert``/``fetch_*``'s
    row-decoding needed adapting, since ``Database.query()`` returns plain
    tuples (positional) rather than ``sqlite3.Row`` (name-addressable).

    Unlike ``SQLiteStore``, this class does not own its connection -- the
    caller opens (and eventually closes) the ``Database`` handle, matching
    DB.2's session-keyed-handle design. This also means this class never
    needs vault credentials itself: whoever already has an open handle
    (e.g. Image Toolkit's unified library session, opened once at login)
    just passes it in.
    """

    def __init__(self, db, table_name: str = "rec_engine_items"):
        self._db = db
        self._table = table_name

    # ------------------------------------------------------------------
    # Connection (no-ops -- the caller owns the handle's lifecycle)
    # ------------------------------------------------------------------

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self) -> bool:
        rows = self._db.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (self._table,),
        )
        existed = len(rows) > 0
        self._db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                id                    TEXT PRIMARY KEY,
                title                 TEXT NOT NULL,
                type                  TEXT,
                watch_status          TEXT,
                rating                REAL,
                year_released         INTEGER,
                num_episodes_or_pages INTEGER,
                genres                TEXT,
                tags                  TEXT,
                associated_entities   TEXT,
                local_file_location   TEXT,
                web_link              TEXT,
                dense_vector          TEXT NOT NULL,
                sparse_indices        TEXT NOT NULL,
                sparse_values         TEXT NOT NULL
            )
            """,
            (),
        )
        if not existed:
            logger.info("Created '%s' table in the unified library store", self._table)
        return not existed

    def collection_info(self) -> dict:
        (count,) = self._db.query(f"SELECT COUNT(*) FROM {self._table}", ())[0]
        return {
            "points_count": count,
            "collection": self._table,
            "storage": "library.db (encrypted, unified store)",
        }

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(self, items: list[EmbeddedItem], batch_size: int = 64) -> int:
        upserted = 0
        placeholders = ", ".join("?" * len(_ALL_COLS))
        sql = (
            f"INSERT OR REPLACE INTO {self._table} ({', '.join(_ALL_COLS)}) "
            f"VALUES ({placeholders})"
        )

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
            if rows:
                self._db.executemany(sql, rows)
            upserted += len(rows)
            logger.debug("Upserted %d/%d rows", upserted, len(items))

        return upserted

    def delete(self, item_id: str) -> None:
        self._db.execute(f"DELETE FROM {self._table} WHERE id = ?", (item_id,))

    # ------------------------------------------------------------------
    # Read helpers used by retriever and pipeline
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        where_clause: str = "",
        params: Optional[list] = None,
    ) -> list[dict]:
        sql = f"SELECT {', '.join(_ALL_COLS)} FROM {self._table}"
        if where_clause:
            sql += f" WHERE {where_clause}"
        rows = self._db.query(sql, tuple(params or []))
        out = []
        for row in rows:
            d = dict(zip(_ALL_COLS, row))
            for col in _JSON_COLS:
                raw = d.get(col)
                d[col] = json.loads(raw) if raw else []
            out.append(d)
        return out

    def fetch_filtered(
        self,
        where_clause: str = "",
        params: Optional[list] = None,
        limit: int = 500,
    ) -> list[dict]:
        sql = f"SELECT {', '.join(_META_COLS)} FROM {self._table}"
        if where_clause:
            sql += f" WHERE {where_clause}"
        sql += f" LIMIT {limit}"
        rows = self._db.query(sql, tuple(params or []))
        out = []
        for row in rows:
            d = dict(zip(_META_COLS, row))
            for col in _META_JSON_COLS:
                raw = d.get(col)
                d[col] = json.loads(raw) if raw else []
            out.append(d)
        return out
