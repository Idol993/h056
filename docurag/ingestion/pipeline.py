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
            )
            return IngestResult(
                success=False,
                failed_files=[file_path.name],
                details=[detail],
                message=detail.error,
            )

        ext = file_path.suffix.lower()
        if ext not in DocumentLoader.SUPPORTED_EXTENSIONS:
            detail.status = "failed"
            detail.error = f"不支持的格式: {ext}"
            self.vector_store.set_file_ingest_state(
                file_path.name, "failed", error=detail.error,
            )
            return IngestResult(
                success=False,
                failed_files=[file_path.name],
                details=[detail],
                message=detail.error,
            )

        try:
            current_hash = compute_file_hash(file_path)
            stored_hash = self.vector_store.get_file_hash(file_path.name)
            detail.old_hash = stored_hash or ""
            detail.new_hash = current_hash

            prev_info = self.vector_store.get_file_info(file_path.name)
            prev_count = prev_info.chunk_count if prev_info else 0
            detail.old_hash = stored_hash or (prev_info.file_hash if prev_info else "")

            if not force and stored_hash and current_hash == stored_hash:
                logger.info(f"文件未变化，跳过: {file_path.name}")
                detail.status = "skipped"
                detail.skipped_chunks = prev_count
                self.vector_store.set_file_ingest_state(
                    file_path.name,
                    "skipped",
                    chunk_count=prev_count,
                    file_hash=current_hash,
                    prev_chunk_count=prev_count,
                    prev_file_hash=stored_hash or "",
                )
                return IngestResult(
                    success=True,
                    skipped_files=[file_path.name],
                    total_skipped=prev_count,
                    details=[detail],
                    message=f"文件未变化，跳过: {file_path.name}",
                )

            if progress_cb:
                progress_cb(f"加载文件: {file_path.name}")

            raw_chunks = self.loader.load_file(file_path)
            if not raw_chunks:
                detail.status = "failed"
                detail.error = f"文件无有效内容: {file_path.name}"
                self.vector_store.set_file_ingest_state(
                    file_path.name, "failed", error=detail.error,
                    prev_chunk_count=prev_count, prev_file_hash=stored_hash or "",
                )
                return IngestResult(
                    success=False,
                    failed_files=[file_path.name],
                    details=[detail],
                    message=detail.error,
                )

            if progress_cb:
                progress_cb(f"切分文本: {file_path.name} ({len(raw_chunks)} 块)")

            split_chunks = self.splitter.split(raw_chunks)
            if not split_chunks:
                detail.status = "failed"
                detail.error = f"文本切分失败: {file_path.name}"
                self.vector_store.set_file_ingest_state(
                    file_path.name, "failed", error=detail.error,
                    prev_chunk_count=prev_count, prev_file_hash=stored_hash or "",
                )
                return IngestResult(
                    success=False,
                    failed_files=[file_path.name],
                    details=[detail],
                    message=detail.error,
                )

            if progress_cb:
                progress_cb(f"向量化: {file_path.name} ({len(split_chunks)} 片段)")

            embeddings = self.embedder.embed_chunks(split_chunks)

            if progress_cb:
                progress_cb(f"去重入库: {file_path.name}")

            self._upsert_file(
                file_path.name, split_chunks, embeddings,
                file_hash=current_hash,
                prev_chunk_count=prev_count,
                prev_file_hash=stored_hash or "",
            )

            new_count = len(split_chunks)
            if prev_count == 0:
                detail.status = "added"
                detail.added_chunks = new_count
            else:
                detail.status = "replaced"
                detail.replaced_chunks = new_count
                detail.removed_chunks = prev_count
            detail.chunk_delta = new_count - prev_count

            self.vector_store.set_file_ingest_state(
                file_path.name,
                detail.status,
                chunk_count=new_count,
                file_hash=current_hash,
                prev_chunk_count=prev_count,
                prev_file_hash=stored_hash or "",
            )

            return IngestResult(
                success=True,
                processed_files=[file_path.name] if detail.status == "added" else [],
                updated_files=[file_path.name] if detail.status == "replaced" else [],
                total_added=detail.added_chunks,
                total_replaced=detail.replaced_chunks,
                total_chunks=new_count,
                details=[detail],
                message=(
                    f"{'新增' if detail.status == 'added' else '更新'}成功: {file_path.name} "
                    f"({new_count} 片段, delta={'+' if detail.chunk_delta >= 0 else ''}{detail.chunk_delta})"
                ),
            )

        except Exception as e:
            logger.exception(f"摄入文件失败 {file_path}: {e}")
            detail.status = "failed"
            detail.error = str(e)
            self.vector_store.set_file_ingest_state(
                file_path.name, "failed", error=detail.error,
            )
            return IngestResult(
                success=False,
                failed_files=[file_path.name],
                details=[detail],
                message=str(e),
            )

    def _upsert_file(
        self,
        filename: str,
        chunks: List[DocumentChunk],
        embeddings: list,
        file_hash: Optional[str] = None,
        prev_chunk_count: int = 0,
        prev_file_hash: str = "",
    ):
        self.vector_store.delete_by_file(filename)
        self.vector_store.add_documents(chunks, embeddings, file_hash=file_hash)
        logger.info(
            f"文件 {filename} 已更新入库 ({len(chunks)} 片段, "
            f"hash={file_hash[:16] if file_hash else 'none'}, "
            f"prev={prev_chunk_count}, prev_hash={prev_file_hash[:16] if prev_file_hash else 'none'})"
        )

    def remove_file(self, filename: str) -> bool:
        try:
            self.vector_store.delete_by_file(filename)
            logger.info(f"已从向量库移除文件: {filename}")
            return True
        except Exception as e:
            logger.error(f"移除文件失败 {filename}: {e}")
            return False

    def list_files(self, upload_dir: Optional[Path] = None) -> List[FileInfo]:
        return self.vector_store.list_files_with_info(upload_dir)

    def get_file_info(self, filename: str, upload_dir: Optional[Path] = None) -> Optional[FileInfo]:
        return self.vector_store.get_file_info(filename, upload_dir)

    def doctor(self, upload_dir: Path = UPLOAD_DIR, watcher_running: bool = False) -> DoctorReport:
        supported = DocumentLoader.SUPPORTED_EXTENSIONS
        actual_files = {}
        if upload_dir.exists():
            for f in upload_dir.iterdir():
                if f.is_file():
                    actual_files[f.name] = f

        db_files = set(self.vector_store.list_files())
        actual_supported = {n for n, p in actual_files.items() if p.suffix.lower() in supported}

        report = DoctorReport(
            upload_dir=str(upload_dir),
            db_dir=self.vector_store.persist_dir,
            watcher_running=watcher_running,
            total_in_uploads=len(actual_supported),
            total_in_db=len(db_files),
        )

        for filename in db_files - actual_supported:
            report.orphan_files.append(filename)

        for filename in actual_supported - db_files:
            report.missing_files.append(filename)

        for filename in actual_supported & db_files:
            info = self.vector_store.get_file_info(filename, upload_dir=upload_dir)
            if info:
                current_hash = compute_file_hash(upload_dir / filename)
                if info.file_hash and current_hash and current_hash != info.file_hash:
                    report.hash_mismatch.append(filename)
                if info.updated_at:
                    mtime = (upload_dir / filename).stat().st_mtime
                    if mtime > info.updated_at + 5:
                        report.stale_files.append(filename)

        for name, path in actual_files.items():
            if path.suffix.lower() not in supported:
                report.unsupported_files.append(name)
            elif path.stat().st_size == 0:
                report.empty_files.append(name)

        report.orphan_files.sort()
        report.missing_files.sort()
        report.stale_files.sort()
        report.hash_mismatch.sort()
        report.empty_files.sort()
        report.unsupported_files.sort()

        return report

    def fix_doctor_issues(self, report: DoctorReport, progress_cb: ProgressCallback = None) -> IngestResult:
        from docurag.config import UPLOAD_DIR
        upload_dir = Path(report.upload_dir) if report.upload_dir else UPLOAD_DIR

        for fn in report.orphan_files:
            if progress_cb:
                progress_cb(f"清理孤儿记录: {fn}")
            self.remove_file(fn)

        result = IngestResult(success=True)
        for fn in report.missing_files + report.hash_mismatch + report.stale_files:
            fp = upload_dir / fn
            if fp.exists():
                r = self.ingest_file(fp, force=True, progress_cb=progress_cb)
                if r.success:
                    if r.updated_files:
                        result.updated_files.extend(r.updated_files)
                    if r.processed_files:
                        result.processed_files.extend(r.processed_files)
                    if r.details:
                        result.details.extend(r.details)
                    result.total_added += r.total_added
                    result.total_replaced += r.total_replaced
                else:
                    result.failed_files.extend(r.failed_files)
                    if r.details:
                        result.details.extend(r.details)

        result.total_chunks = self.vector_store.count()
        result.message = (
            f"修复完成: 新增 {len(result.processed_files)} 个，更新 {len(result.updated_files)} 个，"
            f"失败 {len(result.failed_files)} 个，清理孤儿 {len(report.orphan_files)} 个，"
            f"库中共 {result.total_chunks} 条记录"
        )
        return result
