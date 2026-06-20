import logging
from typing import List, Optional

from docurag.config import ENABLE_RERANKING, RETRIEVAL_TOP_K, RERANK_TOP_K
from docurag.ingestion.embedder import Embedder
from docurag.retrieval.reranker import Reranker
from docurag.retrieval.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Optional[Reranker] = None,
        top_k: int = RETRIEVAL_TOP_K,
        enable_reranking: bool = ENABLE_RERANKING,
        rerank_top_k: int = RERANK_TOP_K
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker if reranker else (Reranker() if enable_reranking else None)
        self.top_k = top_k
        self.enable_reranking = enable_reranking and self.reranker is not None
        self.rerank_top_k = rerank_top_k

    def retrieve(self, query: str, filter_file: Optional[str] = None) -> List[RetrievedChunk]:
        logger.info(f"检索查询: {query[:50]}...")

        query_embedding = self.embedder.embed_query(query)
        search_k = self.top_k if not self.enable_reranking else max(self.top_k, self.rerank_top_k * 2)

        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=search_k,
            filter_file=filter_file
        )

        if not results:
            return []

        if self.enable_reranking:
            logger.info(f"对 {len(results)} 条结果进行重排序")
            results = self.reranker.rerank(query, results, top_k=self.rerank_top_k)

        logger.info(f"检索完成，返回 {len(results)} 条结果")
        return results

    @staticmethod
    def format_sources(chunks: List[RetrievedChunk]) -> List[dict]:
        sources = []
        for chunk in chunks:
            sources.append({
                "file": chunk.source_file,
                "page": chunk.page,
                "snippet": chunk.content[:200]
            })
        return sources
