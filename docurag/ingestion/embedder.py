import logging
from typing import List

import numpy as np

from docurag.config import EMBEDDING_BATCH_SIZE, EMBEDDING_DIMENSION, EMBEDDING_MODEL
from docurag.ingestion.loader import DocumentChunk

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = EMBEDDING_BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            logger.info(f"加载向量化模型: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
        return self._model

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[np.ndarray]:
        if not chunks:
            return []

        model = self._get_model()
        texts = [chunk.content for chunk in chunks]
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            logger.info(f"向量化批次 {i // self.batch_size + 1}: {len(batch_texts)} 条文本")
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            all_embeddings.extend(batch_embeddings)

        logger.info(f"向量化完成: {len(all_embeddings)} 个向量，维度 {EMBEDDING_DIMENSION}")
        return all_embeddings

    def embed_query(self, query: str) -> np.ndarray:
        model = self._get_model()
        return model.encode(query, convert_to_numpy=True, show_progress_bar=False)
