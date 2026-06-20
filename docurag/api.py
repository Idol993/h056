import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from docurag.config import UPLOAD_DIR

logger = logging.getLogger(__name__)

_AUTO_WATCH = True
_AUTO_INGEST_ON_START = True
_watcher = None
_pipeline = None


def set_auto_watch(enabled: bool):
    global _AUTO_WATCH
    _AUTO_WATCH = enabled


def set_auto_ingest_on_start(enabled: bool):
    global _AUTO_INGEST_ON_START
    _AUTO_INGEST_ON_START = enabled


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from docurag.ingestion import IngestionPipeline
        _pipeline = IngestionPipeline()
    return _pipeline


def _start_watcher():
    global _watcher
    if _watcher is not None or not _AUTO_WATCH:
        return
    try:
        from docurag.ingestion.watcher import DirectoryWatcher
        _watcher = DirectoryWatcher(watch_dir=UPLOAD_DIR, pipeline=_get_pipeline())
        _watcher.start(ingest_existing=_AUTO_INGEST_ON_START)
        logger.info("API 服务启动，uploads 目录自动监听已开启")
    except Exception as e:
        logger.warning(f"目录监听器启动失败: {e}")


def _stop_watcher():
    global _watcher
    if _watcher is not None:
        try:
            _watcher.stop()
        except Exception:
            pass
        _watcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _start_watcher()
    yield
    _stop_watcher()


app = FastAPI(title="DocuRAG API", version="0.3.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    filter_file: Optional[str] = None
    filter_ext: Optional[str] = None
    updated_after: Optional[float] = None
    updated_before: Optional[float] = None
    stream: bool = False
    retrieve_only: bool = False
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str = ""
    sources: list[dict] = []


class IngestResponse(BaseModel):
    status: str
    message: str
    total_documents: int
    processed_files: list[str] = []
    updated_files: list[str] = []
    skipped_files: list[str] = []
    removed_files: list[str] = []
    failed_files: list[str] = []
    total_added: int = 0
    total_replaced: int = 0
    total_skipped: int = 0
    details: list[dict] = []


class FileIngestDetailResponse(BaseModel):
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


class FileInfoResponse(BaseModel):
    filename: str
    chunk_count: int
    file_hash: str = ""
    updated_at: float = 0.0
    updated_at_str: str = ""
    exists_in_uploads: bool = False
    last_ingest_status: str = ""
    last_ingest_error: str = ""
    prev_chunk_count: int = 0
    prev_file_hash: str = ""


class DoctorReportResponse(BaseModel):
    upload_dir: str
    db_dir: str
    watcher_running: bool
    total_in_uploads: int
    total_in_db: int
    orphan_files: list[str] = []
    missing_files: list[str] = []
    stale_files: list[str] = []
    empty_files: list[str] = []
    unsupported_files: list[str] = []
    hash_mismatch: list[str] = []


class StatusResponse(BaseModel):
    status: str
    total_documents: int
    files: list[str]
    watching: bool
    upload_dir: str


def _format_time(ts: float) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def _file_info_to_response(info) -> FileInfoResponse:
    return FileInfoResponse(
        filename=info.filename,
        chunk_count=info.chunk_count,
        file_hash=info.file_hash,
        updated_at=info.updated_at,
        updated_at_str=_format_time(info.updated_at),
        exists_in_uploads=info.exists_in_uploads,
        last_ingest_status=info.last_ingest_status,
        last_ingest_error=info.last_ingest_error,
        prev_chunk_count=info.prev_chunk_count,
        prev_file_hash=info.prev_file_hash,
    )


def _detail_to_response(d) -> dict:
    return {
        "filename": d.filename,
        "status": d.status,
        "added_chunks": d.added_chunks,
        "replaced_chunks": d.replaced_chunks,
        "skipped_chunks": d.skipped_chunks,
        "removed_chunks": d.removed_chunks,
        "chunk_delta": d.chunk_delta,
        "old_hash": d.old_hash,
        "new_hash": d.new_hash,
        "error": d.error,
    }


def _get_retriever(top_k: Optional[int] = None):
    from docurag.ingestion import Embedder
    from docurag.retrieval import Retriever, Reranker, VectorStore

    vector_store = VectorStore()
    embedder = Embedder()
    reranker = Reranker()
    kwargs = {}
    if top_k:
        kwargs["top_k"] = top_k
        kwargs["rerank_top_k"] = min(top_k, 5)
    return Retriever(vector_store, embedder, reranker, **kwargs), vector_store


@app.get("/status", response_model=StatusResponse)
def get_status():
    try:
        from docurag.ingestion.loader import DocumentLoader

        pipeline = _get_pipeline()

        supported = DocumentLoader.SUPPORTED_EXTENSIONS
        if UPLOAD_DIR.exists():
            actual_files = {
                f.name for f in UPLOAD_DIR.iterdir()
                if f.is_file() and f.suffix.lower() in supported
            }
        else:
            actual_files = set()

        db_files = set(pipeline.vector_store.list_files())
        orphans = db_files - actual_files
        for fn in orphans:
            pipeline.remove_file(fn)

        clean_files = pipeline.vector_store.list_files()

        return StatusResponse(
            status="ok",
            total_documents=pipeline.vector_store.count(),
            files=sorted(clean_files),
            watching=(_watcher is not None and getattr(_watcher, "running", False)),
            upload_dir=str(UPLOAD_DIR),
        )
    except Exception as e:
        logger.exception(f"获取状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/doctor", response_model=DoctorReportResponse)
def doctor():
    try:
        pipeline = _get_pipeline()
        report = pipeline.doctor(
            upload_dir=UPLOAD_DIR,
            watcher_running=(_watcher is not None and getattr(_watcher, "running", False)),
        )
        return DoctorReportResponse(
            upload_dir=report.upload_dir,
            db_dir=report.db_dir,
            watcher_running=report.watcher_running,
            total_in_uploads=report.total_in_uploads,
            total_in_db=report.total_in_db,
            orphan_files=report.orphan_files,
            missing_files=report.missing_files,
            stale_files=report.stale_files,
            empty_files=report.empty_files,
            unsupported_files=report.unsupported_files,
            hash_mismatch=report.hash_mismatch,
        )
    except Exception as e:
        logger.exception(f"诊断失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/doctor/fix", response_model=IngestResponse)
def doctor_fix():
    try:
        pipeline = _get_pipeline()
        report = pipeline.doctor(
            upload_dir=UPLOAD_DIR,
            watcher_running=(_watcher is not None and getattr(_watcher, "running", False)),
        )
        result = pipeline.fix_doctor_issues(report)
        return IngestResponse(
            status="success" if result.success else "failed",
            message=result.message,
            total_documents=result.total_chunks,
            processed_files=result.processed_files,
            updated_files=result.updated_files,
            skipped_files=result.skipped_files,
            removed_files=report.orphan_files,
            failed_files=result.failed_files,
            total_added=result.total_added,
            total_replaced=result.total_replaced,
            total_skipped=result.total_skipped,
            details=[_detail_to_response(d) for d in result.details],
        )
    except Exception as e:
        logger.exception(f"修复失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest", response_model=IngestResponse)
def ingest_documents(force: bool = Query(False, description="强制重建所有文件")):
    try:
        pipeline = _get_pipeline()
        result = pipeline.sync_directory(UPLOAD_DIR, clear_first=False, force=force)

        if (
            not result.processed_files
            and not result.updated_files
            and not result.skipped_files
            and not result.removed_files
            and not result.failed_files
        ):
            return IngestResponse(
                status="warning",
                message=result.message,
                total_documents=result.total_chunks,
                details=[],
            )

        return IngestResponse(
            status="success" if result.success else "partial",
            message=result.message,
            total_documents=result.total_chunks,
            processed_files=result.processed_files,
            updated_files=result.updated_files,
            skipped_files=result.skipped_files,
            removed_files=result.removed_files,
            failed_files=result.failed_files,
            total_added=result.total_added,
            total_replaced=result.total_replaced,
            total_skipped=result.total_skipped,
            details=[_detail_to_response(d) for d in result.details],
        )
    except Exception as e:
        logger.exception(f"文档同步失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    from docurag.generation import LLMClient, PromptBuilder
    from docurag.retrieval import RetrieveFilters, Retriever

    try:
        retriever, vector_store = _get_retriever(top_k=request.top_k)

        if vector_store.count() == 0:
            raise HTTPException(status_code=400, detail="向量库为空，请先摄入文档")

        filters = RetrieveFilters(
            file_name=request.filter_file,
            file_ext=request.filter_ext,
            updated_after=request.updated_after,
            updated_before=request.updated_before,
        )
        retrieved = retriever.retrieve(request.question, filters=filters)

        if not retrieved:
            return QueryResponse(answer="未找到相关信息", sources=[])

        sources = Retriever.format_sources(retrieved)

        if request.retrieve_only:
            return QueryResponse(answer="", sources=sources)

        if request.stream:
            prompt_builder = PromptBuilder()
            llm_client = LLMClient()
            prompt = prompt_builder.build(request.question, retrieved)

            def generate_stream():
                try:
                    stream = llm_client.generate(prompt, stream=True)
                    if stream:
                        for token in stream:
                            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'done': True, 'sources': sources}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    logger.error(f"流式生成失败: {e}")
                    yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
            )
        else:
            prompt_builder = PromptBuilder()
            llm_client = LLMClient()
            prompt = prompt_builder.build(request.question, retrieved)
            answer = llm_client.generate(prompt, stream=False)

            return QueryResponse(answer=answer or "未找到相关信息", sources=sources)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/docs", response_model=list[FileInfoResponse])
def list_documents():
    try:
        pipeline = _get_pipeline()
        files = pipeline.list_files(upload_dir=UPLOAD_DIR)
        return [_file_info_to_response(f) for f in files]
    except Exception as e:
        logger.exception(f"获取文件列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/docs/{filename}", response_model=FileInfoResponse)
def get_document(filename: str):
    try:
        pipeline = _get_pipeline()
        info = pipeline.get_file_info(filename, upload_dir=UPLOAD_DIR)
        if not info:
            raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
        return _file_info_to_response(info)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"获取文件详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/docs/{filename}/ingest", response_model=IngestResponse)
def reingest_document(filename: str, force: bool = Query(True, description="强制重建")):
    try:
        pipeline = _get_pipeline()
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"文件不存在于 uploads: {filename}")

        result = pipeline.ingest_file(file_path, force=force)
        if not result.success:
            raise HTTPException(status_code=500, detail=result.message)

        return IngestResponse(
            status="success",
            message=result.message,
            total_documents=pipeline.vector_store.count(),
            processed_files=result.processed_files,
            updated_files=result.updated_files,
            skipped_files=result.skipped_files,
            removed_files=[],
            failed_files=result.failed_files,
            total_added=result.total_added,
            total_replaced=result.total_replaced,
            total_skipped=result.total_skipped,
            details=[_detail_to_response(d) for d in result.details],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"重新摄入失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/docs/{filename}")
def delete_document(filename: str):
    try:
        pipeline = _get_pipeline()
        info = pipeline.get_file_info(filename)
        if not info:
            raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

        removed_count = info.chunk_count
        if not pipeline.remove_file(filename):
            raise HTTPException(status_code=500, detail=f"删除失败: {filename}")

        return {
            "status": "success",
            "message": f"已删除文件: {filename}",
            "deleted": filename,
            "removed_chunks": removed_count,
            "total_documents": pipeline.vector_store.count(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"删除文件失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
