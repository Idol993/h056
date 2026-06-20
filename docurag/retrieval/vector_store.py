import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    vector_score: float = 0.0
    rerank_score: Optional[float] = None
    rank: int = 0


@dataclass
class RetrieveDebugInfo:
    filters_applied: dict
    matched_files: List[str]
    total_chunks_before_rerank: int = 0
    total_chunks_after_rerank: int = 0
    vector_top_k: int = 0
    rerank_top_k: int = 0
    enable_reranking: bool = False


@dataclass
class IngestHistoryEntry:
    timestamp: float
    status: str
    chunk_count: int
    chunk_delta: int
    file_hash: str
    prev_file_hash: str
    source: str
    error: str = ""


@dataclass
class FileInfo:
    filename: str
    chunk_count: int
    file_hash: str = ""
    updated_at: float = 0.0
    exists_in_uploads: bool = False
    last_ingest_status: str = ""
    last_ingest_error: str = ""
    prev_chunk_count: int = 0
    prev_file_hash: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class DoctorReport:
    upload_dir: str
    db_dir: str
    watcher_running: bool
    total_in_uploads: int
    total_in_db: int
    orphan_files: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    stale_files: List[str] = field(default_factory=list)
    empty_files: List[str] = field(default_factory=list)
    unsupported_files: List[str] = field(default_factory=list)
    hash_mismatch: List[str] = field(default_factory=list)


def compute_file_hash(file_path: str | Path) -> str:
    path = Path(file_path)
    if not path.exists():
        return ""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
    except Exception:
        return ""
    return h.hexdigest()


_STATE_DIR_NAME = "_docurag_state"
_STATE_FILE_NAME = "ingest_state.json"


class VectorStore:
    def __init__(self, persist_dir: str | Path = CHROMA_DB_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self._state_dir = os.path.join(self.persist_dir, _STATE_DIR_NAME)
        self._state_file = os.path.join(self._state_dir, _STATE_FILE_NAME)
        self._client = None
        self._collection = None
        self._state: Dict[str, Any] = {}
        self._load_state()

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

    def _load_state(self):
        try:
            os.makedirs(self._state_dir, exist_ok=True)
            if os.path.exists(self._state_file):
                with open(self._state_file, "r", encoding="utf-8") as f:
                    self._state = json.load(f)
        except Exception as e:
            logger.warning(f"加载状态文件失败: {e}")
            self._state = {}

    def _save_state(self):
        try:
            os.makedirs(self._state_dir, exist_ok=True)
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存状态文件失败: {e}")

    def set_file_ingest_state(
        self,
        filename: str,
        status: str,
        error: str = "",
        chunk_count: int = 0,
        file_hash: str = "",
        prev_chunk_count: int = 0,
        prev_file_hash: str = "",
        source: str = "manual"
    ):
        ts = time.time()
        self._state.setdefault("files", {})
        self._state["files"][filename] = {
            "status": status,
            "error": error,
            "chunk_count": chunk_count,
            "file_hash": file_hash,
            "prev_chunk_count": prev_chunk_count,
            "prev_file_hash": prev_file_hash,
            "timestamp": ts,
            "source": source,
        }

        self._state.setdefault("history", {})
        self._state["history"].setdefault(filename, [])
        entry = {
            "timestamp": ts,
            "status": status,
            "chunk_count": chunk_count,
            "chunk_delta": chunk_count - prev_chunk_count,
            "file_hash": file_hash,
            "prev_file_hash": prev_file_hash,
            "source": source,
            "error": error,
        }
        self._state["history"][filename].append(entry)
        if len(self._state["history"][filename]) > 50:
            self._state["history"][filename] = self._state["history"][filename][-50:]

        self._save_state()

    def get_file_ingest_state(self, filename: str) -> Optional[dict]:
        return self._state.get("files", {}).get(filename)

    def append_ingest_history(
        self,
        filename: str,
        status: str,
        chunk_count: int = 0,
        prev_chunk_count: int = 0,
        file_hash: str = "",
        prev_file_hash: str = "",
        source: str = "manual",
        error: str = "",
    ):
        ts = time.time()
        self._state.setdefault("history", {})
        self._state["history"].setdefault(filename, [])
        entry = {
            "timestamp": ts,
            "status": status,
            "chunk_count": chunk_count,
            "chunk_delta": chunk_count - prev_chunk_count,
            "file_hash": file_hash,
            "prev_file_hash": prev_file_hash,
            "source": source,
            "error": error,
        }
        self._state["history"][filename].append(entry)
        if len(self._state["history"][filename]) > 50:
            self._state["history"][filename] = self._state["history"][filename][-50:]
        self._save_state()

    def get_ingest_history(self, filename: str, limit: int = 10) -> List[IngestHistoryEntry]:
        raw_list = self._state.get("history", {}).get(filename, [])
        result = []
        for raw in reversed(raw_list[-limit:]):
            result.append(IngestHistoryEntry(
                timestamp=raw.get("timestamp", 0.0),
                status=raw.get("status", ""),
                chunk_count=raw.get("chunk_count", 0),
                chunk_delta=raw.get("chunk_delta", 0),
                file_hash=raw.get("file_hash", ""),
                prev_file_hash=raw.get("prev_file_hash", ""),
                source=raw.get("source", "manual"),
                error=raw.get("error", ""),
            ))
        return result

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
        filter_file: Optional[str] = None,
        filter_ext: Optional[str] = None,
        updated_after: Optional[float] = None,
        updated_before: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        collection = self._get_collection()

        conds = []
        if filter_file:
            conds.append({"source_file": filter_file})
        if filter_ext:
            conds.append({"source_file": {"$contains": filter_ext.lstrip(".")}})
        if updated_after or updated_before:
            ts_cond = {}
            if updated_after:
                ts_cond["$gte"] = float(updated_after)
            if updated_before:
                ts_cond["$lte"] = float(updated_before)
            conds.append({"updated_at": ts_cond})

        if len(conds) == 1:
            where = conds[0]
        elif len(conds) > 1:
            where = {"$and": conds}
        else:
            where = None

        query_vec = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        try:
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                where=where
            )
        except Exception as e:
            logger.warning(f"向量检索 with where 失败，回退到无过滤检索后软件过滤: {e}")
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                where=None
            )

        retrieved = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                src_file = metadata.get("source_file", "")

                if filter_file and src_file != filter_file:
                    continue
                if filter_ext:
                    ext_norm = "." + filter_ext.lower().lstrip(".")
                    if not src_file.lower().endswith(ext_norm):
                        continue
                ts = metadata.get("updated_at", 0)
                if updated_after and ts and float(ts) < float(updated_after):
                    continue
                if updated_before and ts and float(ts) > float(updated_before):
                    continue

                page = metadata.get("page")
                if page == -1:
                    page = None
                vec_score = results["distances"][0][i] if results["distances"] else 0.0
                retrieved.append(RetrievedChunk(
                    content=results["documents"][0][i],
                    source_file=src_file,
                    page=page,
                    score=vec_score,
                    vector_score=vec_score,
                    metadata=metadata
                ))

        logger.info(f"向量检索完成，返回 {len(retrieved)} 条结果 (where={where})")
        return retrieved

    def delete_by_file(self, filename: str):
        collection = self._get_collection()
        results = collection.get(where={"source_file": filename})
        if results and results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"已删除文件 {filename} 的 {len(results['ids'])} 条向量记录")
        if "files" in self._state and filename in self._state["files"]:
            del self._state["files"][filename]
            self._save_state()

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
            ingest_state = self.get_file_ingest_state(filename)
            if ingest_state:
                return FileInfo(
                    filename=filename,
                    chunk_count=0,
                    file_hash="",
                    updated_at=0.0,
                    exists_in_uploads=(upload_dir / filename).exists() if upload_dir else False,
                    last_ingest_status=ingest_state.get("status", ""),
                    last_ingest_error=ingest_state.get("error", ""),
                    prev_chunk_count=ingest_state.get("prev_chunk_count", 0),
                    prev_file_hash=ingest_state.get("prev_file_hash", ""),
                )
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

        ingest_state = self.get_file_ingest_state(filename) or {}

        return FileInfo(
            filename=filename,
            chunk_count=chunk_count,
            file_hash=file_hash,
            updated_at=updated_at,
            exists_in_uploads=exists_in_uploads,
            last_ingest_status=ingest_state.get("status", ""),
            last_ingest_error=ingest_state.get("error", ""),
            prev_chunk_count=ingest_state.get("prev_chunk_count", 0),
            prev_file_hash=ingest_state.get("prev_file_hash", ""),
            metadata=meta_sample
        )

    def list_files_with_info(self, upload_dir: Optional[Path] = None) -> List[FileInfo]:
        all_names = set(self.list_files())

        for fn in self._state.get("files", {}).keys():
            all_names.add(fn)

        if upload_dir is not None and upload_dir.exists():
            from docurag.ingestion.loader import DocumentLoader
            supported = DocumentLoader.SUPPORTED_EXTENSIONS
            for f in upload_dir.iterdir():
                if f.is_file() and f.suffix.lower() in supported:
                    all_names.add(f.name)

        result = []
        for fn in sorted(all_names):
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
        self._state = {}
        self._save_state()
        logger.info("向量库已清空")
