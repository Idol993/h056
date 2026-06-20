import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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


@dataclass
class FileInfo:
    filename: str
    chunk_count: int
    file_hash: str = ""
    updated_at: float = 0.0
    exists_in_uploads: bool = False
    metadata: dict = field(default_factory=dict)


def compute_file_hash(file_path: str | Path) -> str:
    path = Path(file_path)
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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

    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[np.ndarray],
        file_hash: Optional[str] = None,
        updated_at: Optional[float] = None
    ):
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks 数量 ({len(chunks)}) 与 embeddings 数量 ({len(embeddings)}) 不匹配")

        collection = self._get_collection()
        ts = updated_at if updated_at is not None else time.time()

        ids = []
        documents = []
        metadatas = []
        vectors = []

        for chunk, embedding in zip(chunks, embeddings):
            doc_id = str(uuid.uuid4())
            metadata = {
                "source_file": chunk.source_file,
                "page": chunk.page if chunk.page is not None else -1,
                "updated_at": float(ts),
            }
            if file_hash:
                metadata["file_hash"] = file_hash
            if chunk.metadata:
                for k, v in chunk.metadata.items():
                    if k not in metadata and isinstance(v, (str, int, float, bool)):
                        metadata[k] = v

            ids.append(doc_id)
            documents.append(chunk.content)
            metadatas.append(metadata)
            vectors.append(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)

        logger.info(f"向向量库添加 {len(ids)} 条文档 (file_hash={file_hash[:16] if file_hash else 'none'})")
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

    def get_file_hash(self, filename: str) -> Optional[str]:
        collection = self._get_collection()
        results = collection.get(
            where={"source_file": filename},
            limit=1
        )
        if results and results["metadatas"]:
            meta = results["metadatas"][0]
            return meta.get("file_hash")
        return None

    def get_file_updated_at(self, filename: str) -> Optional[float]:
        collection = self._get_collection()
        results = collection.get(
            where={"source_file": filename},
            limit=1
        )
        if results and results["metadatas"]:
            meta = results["metadatas"][0]
            ts = meta.get("updated_at")
            return float(ts) if ts is not None else None
        return None

    def get_file_info(self, filename: str, upload_dir: Optional[Path] = None) -> Optional[FileInfo]:
        collection = self._get_collection()
        results = collection.get(where={"source_file": filename})
        if not results or not results["ids"]:
            return None

        chunk_count = len(results["ids"])
        file_hash = ""
        updated_at = 0.0
        meta_sample = {}

        if results["metadatas"]:
            meta = results["metadatas"][0]
            file_hash = meta.get("file_hash", "")
            ts = meta.get("updated_at")
            updated_at = float(ts) if ts is not None else 0.0
            meta_sample = {k: v for k, v in meta.items() if k not in ("source_file", "page", "file_hash", "updated_at")}

        exists_in_uploads = False
        if upload_dir:
            exists_in_uploads = (upload_dir / filename).exists()

        return FileInfo(
            filename=filename,
            chunk_count=chunk_count,
            file_hash=file_hash,
            updated_at=updated_at,
            exists_in_uploads=exists_in_uploads,
            metadata=meta_sample
        )

    def list_files_with_info(self, upload_dir: Optional[Path] = None) -> List[FileInfo]:
        filenames = self.list_files()
        result = []
        for fn in filenames:
            info = self.get_file_info(fn, upload_dir)
            if info:
                result.append(info)
        return result

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
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None
        logger.info("向量库已清空")
