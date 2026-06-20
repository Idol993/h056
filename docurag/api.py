import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from docurag.config import UPLOAD_DIR

logger = logging.getLogger(__name__)

_AUTO_WATCH = True
_watcher = None
_pipeline = None


def set_auto_watch(enabled: bool):
    global _AUTO_WATCH
    _AUTO_WATCH = enabled


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
        _watcher.start()
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


app = FastAPI(title="DocuRAG API", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    filter_file: Optional[str] = None
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class IngestResponse(BaseModel):
    status: str
    message: str
    total_documents: int
    processed_files: list[str] = []
    skipped_files: list[str] = []


class StatusResponse(BaseModel):
    status: str
    total_documents: int
    files: list[str]
    watching: bool


def _get_retriever():
    from docurag.ingestion import Embedder
    from docurag.retrieval import Retriever, Reranker, VectorStore

    vector_store = VectorStore()
    embedder = Embedder()
    reranker = Reranker()
    return Retriever(vector_store, embedder, reranker), vector_store


@app.get("/status", response_model=StatusResponse)
def get_status():
    try:
        _, vector_store = _get_retriever()
        return StatusResponse(
            status="ok",
            total_documents=vector_store.count(),
            files=vector_store.list_files(),
            watching=(_watcher is not None and _watcher._running)
        )
    except Exception as e:
        logger.exception(f"获取状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest", response_model=IngestResponse)
def ingest_documents():
    try:
        pipeline = _get_pipeline()
        result = pipeline.ingest_directory(UPLOAD_DIR, clear_first=False)

        if not result.processed_files and not result.skipped_files:
            return IngestResponse(
                status="warning",
                message=result.message,
                total_documents=result.total_chunks,
                processed_files=[],
                skipped_files=[]
            )

        return IngestResponse(
            status="success" if result.success else "partial",
            message=result.message,
            total_documents=result.total_chunks,
            processed_files=result.processed_files,
            skipped_files=result.skipped_files
        )
    except Exception as e:
        logger.exception(f"文档摄入失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    from docurag.generation import LLMClient, PromptBuilder
    from docurag.retrieval import Retriever

    try:
        retriever, vector_store = _get_retriever()

        if vector_store.count() == 0:
            raise HTTPException(status_code=400, detail="向量库为空，请先摄入文档")

        retrieved = retriever.retrieve(request.question, filter_file=request.filter_file)

        if not retrieved:
            return QueryResponse(answer="未找到相关信息", sources=[])

        sources = Retriever.format_sources(retrieved)

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
                media_type="text/event-stream"
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
