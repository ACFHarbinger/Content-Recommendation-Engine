"""
Phase 1 — BGE-M3 embedding wrapper.

Produces dense (1024-dim) and sparse (SPLADE-style) vectors from text.
The model is loaded once as a module-level singleton (thread-safe after init).
"""

from __future__ import annotations

import logging
import threading
import time

from src.core.schema import EmbeddedItem, MediaItem

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def _get_model(model_name: str = "BAAI/bge-m3"):
    """Lazy-load and cache the BGE-M3 model as a process-level singleton."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise RuntimeError(
                "FlagEmbedding is not installed. "
                "Run: pip install 'FlagEmbedding>=1.3.5'"
            ) from e
        t0 = time.perf_counter()
        logger.info("Loading %s — first load may download ~2 GB…", model_name)
        _model = BGEM3FlagModel(model_name, use_fp16=True)
        logger.info("Model loaded in %.1fs", time.perf_counter() - t0)
        return _model


class Embedder:
    """
    Wraps BGEM3FlagModel with a clean API for the recommendation pipeline.

    Methods
    -------
    embed_dense(text)
        Returns a 1024-dim float list for semantic search.
    embed_sparse(text)
        Returns (indices, values) for lexical/SPLADE search.
    embed_batch(items)
        Embeds a batch of MediaItems, returning EmbeddedItem list.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", max_length: int = 512):
        self._model_name = model_name
        self._max_length = max_length

    @property
    def _model(self):
        return _get_model(self._model_name)

    def embed_dense(self, text: str) -> list[float]:
        output = self._model.encode(
            [text],
            max_length=self._max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"][0].tolist()

    def embed_sparse(self, text: str) -> tuple[list[int], list[float]]:
        output = self._model.encode(
            [text],
            max_length=self._max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        sw: dict[str, float] = output["lexical_weights"][0]
        indices = [int(k) for k in sw.keys()]
        values = [v for v in sw.values()]
        return indices, values

    def embed_batch(
        self,
        items: list[MediaItem],
        batch_size: int = 32,
        progress_callback=None,
    ) -> list[EmbeddedItem]:
        """Embed a list of MediaItems in batches; returns EmbeddedItem list."""
        results: list[EmbeddedItem] = []
        total = len(items)

        for start in range(0, total, batch_size):
            batch = items[start : start + batch_size]
            dense_texts = [item.dense_text for item in batch]
            sparse_texts = [item.sparse_text for item in batch]

            dense_out = self._model.encode(
                dense_texts,
                batch_size=batch_size,
                max_length=self._max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            sparse_out = self._model.encode(
                sparse_texts,
                batch_size=batch_size,
                max_length=256,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
            )

            for i, item in enumerate(batch):
                sw: dict[str, float] = sparse_out["lexical_weights"][i]
                results.append(
                    EmbeddedItem(
                        item=item,
                        dense_vector=dense_out["dense_vecs"][i].tolist(),
                        sparse_indices=[int(k) for k in sw.keys()],
                        sparse_values=[v for v in sw.values()],
                    )
                )

            if progress_callback:
                progress_callback(min(start + batch_size, total), total)

            logger.debug("Embedded %d/%d items", min(start + batch_size, total), total)

        return results
