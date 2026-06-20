import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "docurag.log"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DIMENSION = 384

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ENABLE_RERANKING = True

RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_TIMEOUT = 60

CHROMA_COLLECTION_NAME = "docurag_documents"

PROMPT_TEMPLATE = (
    "根据以下参考文档回答问题，如果参考文档中没有答案请回答'未找到相关信息'。\n\n"
    "参考文档：\n{retrieved_docs}\n\n"
    "问题：{question}\n\n"
    "答案："
)

LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5
