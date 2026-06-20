import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from docurag.config import UPLOAD_DIR

logger = logging.getLogger(__name__)

app = FastAPI(title="DocuRAG API", version="0.1.0")


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


class StatusResponse(BaseModel):
    status: str
    total_documents: int
    files: list[str]


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
            files=vector_store.list_files()
        )
    except Exception as e:
        logger.exception(f"获取状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest", response_model=IngestResponse)
def ingest_documents():
    from docurag.ingestion import DocumentLoader, Embedder, TextSplitter
    from docurag.retrieval import VectorStore

    try:
        loader = DocumentLoader()
        splitter = TextSplitter()
        embedder = Embedder()
        vector_store = VectorStore()

        raw_chunks = loader.load_directory(UPLOAD_DIR)
        if not raw_chunks:
            return IngestResponse(
                status="warning",
                message=f"目录 {UPLOAD_DIR} 中没有找到可摄入的文档",
                total_documents=vector_store.count()
            )

        split_chunks = splitter.split(raw_chunks)
        embeddings = embedder.embed_chunks(split_chunks)
        vector_store.add_documents(split_chunks, embeddings)

        total = vector_store.count()
        return IngestResponse(
            status="success",
            message=f"成功摄入 {len(split_chunks)} 个文本片段",
            total_documents=total
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
                full_answer = ""
                try:
                    stream = llm_client.generate(prompt, stream=True)
                    if stream:
                        for token in stream:
                            full_answer += token
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
