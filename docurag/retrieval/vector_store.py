import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
import numpy as np

from docurag.config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR
from docurag.ingestion.loader import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    content: str
    source_file: str
    page: Optional[int]
    score: float
    metadata: dict


class VectorStore:
    def __init__(self, persist_dir: str | Path = CHROMA_DB_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _get_client(self):
        if self._client is None:
            logger.info(f"初始化 ChromaDB，持久化目录: {self.persist_dir}")
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def count(self) -> int:
        collection = self._get_collection()
        return collection.count()

    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[np.ndarray]):
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks 数量 ({len(chunks)}) 与 embeddings 数量 ({len(embeddings)}) 不匹配")

        collection = self._get_collection()

        ids = []
        documents = []
        metadatas = []
        vectors = []

        for chunk, embedding in zip(chunks, embeddings):
            doc_id = str(uuid.uuid4())
            metadata = {
                "source_file": chunk.source_file,
                "page": chunk.page if chunk.page is not None else -1,
            }
            if chunk.metadata:
                for k, v in chunk.metadata.items():
                    if k not in metadata and isinstance(v, (str, int, float, bool)):
                        metadata[k] = v

            ids.append(doc_id)
            documents.append(chunk.content)
            metadatas.append(metadata)
            vectors.append(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)

        logger.info(f"向向量库添加 {len(ids)} 条文档")
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vectors
        )

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_file: Optional[str] = None
    ) -> List[RetrievedChunk]:
        collection = self._get_collection()
        where = {"source_file": filter_file} if filter_file else None

        query_vec = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where
        )

        retrieved = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                page = metadata.get("page")
                if page == -1:
                    page = None
                retrieved.append(RetrievedChunk(
                    content=results["documents"][0][i],
                    source_file=metadata.get("source_file", "unknown"),
                    page=page,
                    score=results["distances"][0][i] if results["distances"] else 0.0,
                    metadata=metadata
                ))

        logger.info(f"向量检索完成，返回 {len(retrieved)} 条结果")
        return retrieved

    def delete_by_file(self, filename: str):
        collection = self._get_collection()
        results = collection.get(where={"source_file": filename})
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"已删除文件 {filename} 的 {len(results['ids'])} 条向量记录")

    def list_files(self) -> List[str]:
        collection = self._get_collection()
        results = collection.get()
        if not results or not results["metadatas"]:
            return []
        files = set()
        for meta in results["metadatas"]:
            if meta and "source_file" in meta:
                files.add(meta["source_file"])
        return sorted(list(files))

    def clear(self):
        client = self._get_client()
        client.delete_collection(self.collection_name)
        self._collection = None
        logger.info("向量库已清空")
