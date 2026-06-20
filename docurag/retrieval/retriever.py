import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from docurag.config import ENABLE_RERANKING, RETRIEVAL_TOP_K, RERANK_TOP_K
from docurag.ingestion.embedder import Embedder
from docurag.retrieval.reranker import Reranker
from docurag.retrieval.vector_store import RetrieveDebugInfo, RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrieveFilters:
    file_name: Optional[str] = None
    file_ext: Optional[str] = None
    updated_after: Optional[float] = None
    updated_before: Optional[float] = None


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Optional[Reranker] = None,
        top_k: int = RETRIEVAL_TOP_K,
        enable_reranking: bool = ENABLE_RERANKING,
        rerank_top_k: int = RERANK_TOP_K,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker if reranker else (Reranker() if enable_reranking else None)
        self.top_k = top_k
        self.enable_reranking = enable_reranking and self.reranker is not None
        self.rerank_top_k = rerank_top_k

    def retrieve(
        self,
        query: str,
        filter_file: Optional[str] = None,
        filters: Optional[RetrieveFilters] = None,
        return_debug: bool = False,
    ) -> Union[List[RetrievedChunk], Tuple[List[RetrievedChunk], RetrieveDebugInfo]]:
        logger.info(f"检索查询: {query[:50]}...")

        if filters is None:
            filters = RetrieveFilters(file_name=filter_file)
        elif filter_file and not filters.file_name:
            filters.file_name = filter_file

        query_embedding = self.embedder.embed_query(query)
        search_k = self.top_k if not self.enable_reranking else max(self.top_k, self.rerank_top_k * 2)

        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=search_k,
            filter_file=filters.file_name,
            filter_ext=filters.file_ext,
            updated_after=filters.updated_after,
            updated_before=filters.updated_before,
        )

        debug = RetrieveDebugInfo(
            filters_applied={
                "file_name": filters.file_name,
                "file_ext": filters.file_ext,
                "updated_after": filters.updated_after,
                "updated_before": filters.updated_before,
            },
            matched_files=sorted(list({c.source_file for c in results})),
            total_chunks_before_rerank=len(results),
            vector_top_k=search_k,
            rerank_top_k=self.rerank_top_k,
            enable_reranking=self.enable_reranking,
        )

        if not results:
            if return_debug:
                return [], debug
            return []

        before_rerank_count = len(results)

        if self.enable_reranking and len(results) > 1:
            logger.info(f"对 {len(results)} 条结果进行重排序")
            raw_before = {id(c): c for c in results}
            results = self.reranker.rerank(query, results, top_k=self.rerank_top_k)
            for idx, c in enumerate(results):
                c.rerank_score = c.score
                c.rank = idx + 1
        else:
            for idx, c in enumerate(results):
                c.rank = idx + 1

        debug.total_chunks_before_rerank = before_rerank_count
        debug.total_chunks_after_rerank = len(results)
        debug.matched_files = sorted(list({c.source_file for c in results}))

        logger.info(f"检索完成，返回 {len(results)} 条结果")

        if return_debug:
            return results, debug
        return results

    @staticmethod
    def format_sources(chunks: List[RetrievedChunk]) -> List[dict]:
        sources = []
        for chunk in chunks:
            entry = {
                "file": chunk.source_file,
                "page": chunk.page,
                "snippet": chunk.content[:200],
                "vector_score": round(chunk.vector_score, 6) if chunk.vector_score else None,
                "rerank_score": round(chunk.rerank_score, 6) if chunk.rerank_score else None,
                "rank": chunk.rank,
            }
            sources.append(entry)
        return sources

    @staticmethod
    def debug_to_dict(debug: RetrieveDebugInfo) -> dict:
        return {
            "filters_applied": debug.filters_applied,
            "matched_files": debug.matched_files,
            "total_chunks_before_rerank": debug.total_chunks_before_rerank,
            "total_chunks_after_rerank": debug.total_chunks_after_rerank,
            "vector_top_k": debug.vector_top_k,
            "rerank_top_k": debug.rerank_top_k,
            "enable_reranking": debug.enable_reranking,
        }
