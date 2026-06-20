import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from docurag.config import UPLOAD_DIR
from docurag.ingestion import DocumentLoader, Embedder, TextSplitter
from docurag.ingestion.loader import DocumentChunk
from docurag.retrieval import VectorStore
from docurag.retrieval.vector_store import (
    DoctorReport,
    FileInfo,
    compute_file_hash,
)

logger = logging.getLogger(__name__)


@dataclass
class FileIngestDetail:
    filename: str
    status: str
    added_chunks: int = 0
    replaced_chunks: int = 0
    skipped_chunks: int = 0
    removed_chunks: int = 0
    chunk_delta: int = 0
    old_hash: str = ""
    new_hash: str = ""
    error: str = ""


@dataclass
class IngestResult:
    success: bool
    processed_files: List[str] = field(default_factory=list)
    updated_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    removed_files: List[str] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    total_chunks: int = 0
    total_added: int = 0
    total_replaced: int = 0
    total_skipped: int = 0
    details: List[FileIngestDetail] = field(default_factory=list)
    message: str = ""


ProgressCallback = Optional[Callable[[str], None]]


class IngestionPipeline:
    def __init__(self):
        self._loader: Optional[DocumentLoader] = None
        self._splitter: Optional[TextSplitter] = None
        self._embedder: Optional[Embedder] = None
        self._vector_store: Optional[VectorStore] = None

    @property
    def loader(self) -> DocumentLoader:
        if self._loader is None:
            self._loader = DocumentLoader()
        return self._loader

    @property
    def splitter(self) -> TextSplitter:
        if self._splitter is None:
            self._splitter = TextSplitter()
        return self._splitter

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    def sync_directory(
        self,
        directory: str | Path = UPLOAD_DIR,
        clear_first: bool = False,
        force: bool = False,
        progress_cb: ProgressCallback = None,
    ) -> IngestResult:
        directory = Path(directory)
        if not directory.is_dir():
            return IngestResult(
                success=False,
                total_chunks=0,
                message=f"目录不存在: {directory}",
            )

        if clear_first:
            self.vector_store.clear()
            if progress_cb:
                progress_cb("已清空向量库")

        supported = DocumentLoader.SUPPORTED_EXTENSIONS
        actual_files = {
            f.name for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in supported
        }
        db_files = set(self.vector_store.list_files())

        removed: List[str] = []
        missing_files = db_files - actual_files
        if missing_files:
            for fn in sorted(missing_files):
                if progress_cb:
                    progress_cb(f"清理已删除文件: {fn}")
                info = self.vector_store.get_file_info(fn, upload_dir=directory)
                prev_count = info.chunk_count if info else 0
                self.vector_store.delete_by_file(fn)
                removed.append(fn)

            if progress_cb:
                progress_cb(f"已清理 {len(removed)} 个已删除文件的记录")

        files_to_process = sorted([directory / fn for fn in actual_files])

        if not files_to_process and not removed:
            return IngestResult(
                success=True,
                processed_files=[],
                updated_files=[],
                skipped_files=[],
                removed_files=removed,
                total_chunks=self.vector_store.count(),
                message=f"目录 {directory} 中没有可摄入的文档",
            )

        processed: List[str] = []
        updated: List[str] = []
        skipped: List[str] = []
        failed: List[str] = []
        details: List[FileIngestDetail] = []
        total_added = 0
        total_replaced = 0
        total_skipped = 0

        for file_path in files_to_process:
            result = self.ingest_file(file_path, force=force, progress_cb=progress_cb)
            if result.success:
                if result.details:
                    d = result.details[0]
                    details.append(d)
                    if d.status == "skipped":
                        skipped.append(file_path.name)
                        total_skipped += d.skipped_chunks
                    elif d.status == "added":
                        processed.append(file_path.name)
                        total_added += d.added_chunks
                    elif d.status == "replaced":
                        updated.append(file_path.name)
                        total_replaced += d.replaced_chunks
                    else:
                        processed.append(file_path.name)
                        total_added += d.added_chunks
                else:
                    processed.append(file_path.name)
            else:
                failed.append(file_path.name)
                if result.details:
                    details.extend(result.details)
                logger.warning(f"跳过文件 {file_path.name}: {result.message}")

        return IngestResult(
            success=True,
            processed_files=processed,
            updated_files=updated,
            skipped_files=skipped,
            removed_files=removed,
            failed_files=failed,
            total_chunks=self.vector_store.count(),
            total_added=total_added,
            total_replaced=total_replaced,
            total_skipped=total_skipped,
            details=details,
            message=(
                f"同步完成: 新增 {len(processed)} 个，更新 {len(updated)} 个，跳过 {len(skipped)} 个，"
                f"失败 {len(failed)} 个，清理 {len(removed)} 个，库中共 {self.vector_store.count()} 条记录"
            ),
        )

    def ingest_directory(
        self,
        directory: str | Path = UPLOAD_DIR,
        clear_first: bool = False,
        force: bool = False,
        progress_cb: ProgressCallback = None,
    ) -> IngestResult:
        return self.sync_directory(directory, clear_first, force, progress_cb)

    def ingest_file(
        self,
        file_path: str | Path,
        force: bool = False,
        progress_cb: ProgressCallback = None,
    ) -> IngestResult:
        file_path = Path(file_path)
        detail = FileIngestDetail(filename=file_path.name)

        if not file_path.exists() or not file_path.is_file():
            detail.status = "failed"
            detail.error = f"文件不存在: {file_path}"
            self.vector_store.set_file_ingest_state(
                file_path.name, "failed", error=detail.error,
