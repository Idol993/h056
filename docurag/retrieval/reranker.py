import logging
from typing import List

from docurag.config import RERANKER_MODEL
from docurag.retrieval.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            logger.info(f"加载重排序模型: {self.model_name}")
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
        return self._model

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int = 5) -> List[RetrievedChunk]:
        if not chunks:
            return []

        if len(chunks) <= top_k:
            return chunks

        try:
            model = self._get_model()
            pairs = [(query, chunk.content) for chunk in chunks]
            scores = model.predict(pairs)

            scored = list(zip(chunks, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for chunk, score in scored[:top_k]:
                reranked.append(RetrievedChunk(
                    content=chunk.content,
                    source_file=chunk.source_file,
                    page=chunk.page,
                    score=float(score),
                    metadata=chunk.metadata
                ))

            logger.info(f"重排序完成: Top-{len(reranked)} / 原始 {len(chunks)}")
            return reranked
        except Exception as e:
            logger.warning(f"重排序失败，返回原始结果: {e}")
            return chunks[:top_k]
