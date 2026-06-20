import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from docurag.config import UPLOAD_DIR
from docurag.ingestion import DocumentLoader, Embedder, TextSplitter
from docurag.ingestion.loader import DocumentChunk
from docurag.retrieval import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    success: bool
    processed_files: List[str]
    skipped_files: List[str]
    total_chunks: int
    message: str


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

    def ingest_directory(
        self,
        directory: str | Path = UPLOAD_DIR,
        clear_first: bool = False,
        progress_cb: ProgressCallback = None
    ) -> IngestResult:
        directory = Path(directory)
        if not directory.is_dir():
            return IngestResult(
                success=False,
                processed_files=[],
                skipped_files=[],
                total_chunks=0,
                message=f"目录不存在: {directory}"
            )

        if clear_first:
            self.vector_store.clear()
            if progress_cb:
                progress_cb("已清空向量库")

        supported = DocumentLoader.SUPPORTED_EXTENSIONS
        files_to_process = [
            f for f in sorted(directory.iterdir())
            if f.is_file() and f.suffix.lower() in supported
        ]

        if not files_to_process:
            return IngestResult(
                success=True,
                processed_files=[],
                skipped_files=[],
                total_chunks=self.vector_store.count(),
                message=f"目录 {directory} 中没有可摄入的文档"
            )

        processed: List[str] = []
        skipped: List[str] = []
        grand_total = 0

        for file_path in files_to_process:
            result = self.ingest_file(file_path, progress_cb=progress_cb)
            if result.success:
                processed.append(file_path.name)
                grand_total += result.total_chunks
            else:
                skipped.append(file_path.name)
                logger.warning(f"跳过文件 {file_path.name}: {result.message}")

        return IngestResult(
            success=True,
            processed_files=processed,
            skipped_files=skipped,
            total_chunks=self.vector_store.count(),
            message=f"处理完成: 成功 {len(processed)} 个文件，跳过 {len(skipped)} 个，库中共 {self.vector_store.count()} 条记录"
        )

    def ingest_file(
        self,
        file_path: str | Path,
        progress_cb: ProgressCallback = None
    ) -> IngestResult:
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            return IngestResult(
                success=False,
                processed_files=[],
                skipped_files=[file_path.name] if file_path.exists() else [],
                total_chunks=0,
                message=f"文件不存在: {file_path}"
            )

        ext = file_path.suffix.lower()
        if ext not in DocumentLoader.SUPPORTED_EXTENSIONS:
            return IngestResult(
                success=False,
                processed_files=[],
                skipped_files=[file_path.name],
                total_chunks=0,
                message=f"不支持的格式: {ext}"
            )

        try:
            if progress_cb:
                progress_cb(f"加载文件: {file_path.name}")

            raw_chunks = self.loader.load_file(file_path)
            if not raw_chunks:
                return IngestResult(
                    success=False,
                    processed_files=[],
                    skipped_files=[file_path.name],
                    total_chunks=0,
                    message=f"文件无有效内容: {file_path.name}"
                )

            if progress_cb:
                progress_cb(f"切分文本: {file_path.name} ({len(raw_chunks)} 块)")

            split_chunks = self.splitter.split(raw_chunks)
            if not split_chunks:
                return IngestResult(
                    success=False,
                    processed_files=[],
                    skipped_files=[file_path.name],
                    total_chunks=0,
                    message=f"文本切分失败: {file_path.name}"
                )

            if progress_cb:
                progress_cb(f"向量化: {file_path.name} ({len(split_chunks)} 片段)")

            embeddings = self.embedder.embed_chunks(split_chunks)

            if progress_cb:
                progress_cb(f"去重入库: {file_path.name}")

            self._upsert_file(file_path.name, split_chunks, embeddings)

            return IngestResult(
                success=True,
                processed_files=[file_path.name],
                skipped_files=[],
                total_chunks=len(split_chunks),
                message=f"摄入成功: {file_path.name} ({len(split_chunks)} 片段)"
            )

        except Exception as e:
            logger.exception(f"摄入文件失败 {file_path}: {e}")
            return IngestResult(
                success=False,
                processed_files=[],
                skipped_files=[file_path.name],
                total_chunks=0,
                message=str(e)
            )

    def _upsert_file(
        self,
        filename: str,
        chunks: List[DocumentChunk],
        embeddings: list
    ):
        self.vector_store.delete_by_file(filename)
        self.vector_store.add_documents(chunks, embeddings)
        logger.info(f"文件 {filename} 已更新入库 ({len(chunks)} 片段)")

    def remove_file(self, filename: str) -> bool:
        try:
            self.vector_store.delete_by_file(filename)
            logger.info(f"已从向量库移除文件: {filename}")
            return True
        except Exception as e:
            logger.error(f"移除文件失败 {filename}: {e}")
            return False
