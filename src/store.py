"""
Phase 1 — Qdrant collection management and upsert.

QdrantStore wraps qdrant-client with the exact multi-vector schema required
by the recommendation engine: one dense cosine field and one sparse dot-product
field per document, plus indexed scalar payload for fast filtered queries.
"""
from __future__ import annotations

import logging
from typing import Optional

from .config import Settings, get_settings
from .schema import EmbeddedItem

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Manages the Qdrant collection for media-item embeddings.

    Usage
    -----
    store = QdrantStore()
    store.create_collection()          # idempotent
    store.upsert(embedded_items)
    info = store.collection_info()
    """

    DENSE_DIM = 1024

    def __init__(self, settings: Optional[Settings] = None):
        self._cfg = settings or get_settings()
        self._client = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as e:
                raise RuntimeError(
                    "qdrant-client not installed. "
                    "Run: pip install 'qdrant-client>=1.9.0'"
                ) from e

            if self._cfg.use_local_qdrant:
                self._client = QdrantClient(path=self._cfg.qdrant_storage_path)
                logger.info(
                    "Connected to local Qdrant at %s", self._cfg.qdrant_storage_path
                )
            else:
                self._client = QdrantClient(
                    url=self._cfg.qdrant_url,
                    api_key=self._cfg.qdrant_api_key,
                )
                logger.info("Connected to remote Qdrant at %s", self._cfg.qdrant_url)
        return self._client

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self) -> bool:
        """
        Create the listings collection if it does not exist.  Idempotent.

        Returns True when the collection was created, False when it already existed.
        """
        from qdrant_client.models import (
            Distance,
            FieldCondition,
            PayloadSchemaType,
            SparseIndexParams,
            SparseVectorParams,
            VectorParams,
        )

        client = self._get_client()
        collection = self._cfg.qdrant_collection
        existing = {c.name for c in client.get_collections().collections}

        if collection in existing:
            logger.debug("Collection '%s' already exists.", collection)
            return False

        client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": VectorParams(
                    size=self.DENSE_DIM, distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )

        # Create payload indexes for filterable scalar fields
        for field, schema_type in [
            ("type", PayloadSchemaType.KEYWORD),
            ("watch_status", PayloadSchemaType.KEYWORD),
            ("genres", PayloadSchemaType.KEYWORD),
            ("tags", PayloadSchemaType.KEYWORD),
            ("rating", PayloadSchemaType.FLOAT),
            ("year_released", PayloadSchemaType.INTEGER),
            ("num_episodes_or_pages", PayloadSchemaType.INTEGER),
        ]:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=schema_type,
            )

        logger.info("Created collection '%s' with payload indexes.", collection)
        return True

    def collection_info(self) -> dict:
        """Return point count and vector configuration."""
        client = self._get_client()
        info = client.get_collection(self._cfg.qdrant_collection)
        return {
            "points_count": info.points_count,
            "collection": self._cfg.qdrant_collection,
            "storage": self._cfg.qdrant_storage_path or self._cfg.qdrant_url,
        }

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(self, items: list[EmbeddedItem], batch_size: int = 64) -> int:
        """
        Upsert a list of EmbeddedItems to Qdrant.

        Returns the number of points successfully upserted.
        """
        from qdrant_client.models import PointStruct, SparseVector

        client = self._get_client()
        collection = self._cfg.qdrant_collection
        upserted = 0

        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            structs = []
            for ei in batch:
                item = ei.item
                payload = {
                    "id": item.id,
                    "title": item.title,
                    "type": item.type or "",
                    "watch_status": item.watch_status or "",
                    "rating": item.rating or 0.0,
                    "year_released": item.year_released or 0,
                    "num_episodes_or_pages": item.num_episodes_or_pages or 0,
                    "genres": item.genres,
                    "tags": item.tags,
                    "associated_entities": item.associated_entities,
                    "local_file_location": item.local_file_location or "",
                    "web_link": item.web_link or "",
                }
                structs.append(
                    PointStruct(
                        id=item.id,
                        vector={
                            "dense": ei.dense_vector,
                            "sparse": SparseVector(
                                indices=ei.sparse_indices,
                                values=ei.sparse_values,
                            ),
                        },
                        payload=payload,
                    )
                )
            client.upsert(collection_name=collection, points=structs)
            upserted += len(structs)
            logger.debug("Upserted %d/%d points", upserted, len(items))

        return upserted

    def delete(self, item_id: str) -> None:
        from qdrant_client.models import PointIdsList

        client = self._get_client()
        client.delete(
            collection_name=self._cfg.qdrant_collection,
            points_selector=PointIdsList(points=[item_id]),
        )
