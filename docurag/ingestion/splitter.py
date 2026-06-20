import logging
from typing import List

from docurag.config import CHUNK_OVERLAP, CHUNK_SIZE
from docurag.ingestion.loader import DocumentChunk

logger = logging.getLogger(__name__)


class TextSplitter:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        try:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
        except Exception as e:
            logger.warning(f"NLTK 数据初始化失败，将使用简单切分: {e}")

    def split(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        result = []
        for chunk in chunks:
            split_chunks = self._split_single_chunk(chunk)
            result.extend(split_chunks)
        logger.info(f"文本切分完成: {len(chunks)} 个文档块切分为 {len(result)} 个片段")
        return result

    def _split_single_chunk(self, chunk: DocumentChunk) -> List[DocumentChunk]:
        sentences = self._split_sentences(chunk.content)
        if not sentences:
            return []

        result = []
        current_parts = []
        current_length = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_len = len(sentence)

            if current_length + sentence_len + 1 <= self.chunk_size:
                current_parts.append(sentence)
                current_length += sentence_len + 1
                i += 1
            else:
                if current_parts:
                    result.append(self._build_chunk(chunk, current_parts))
                    overlap_parts = self._get_overlap(current_parts)
                    current_parts = overlap_parts
                    current_length = sum(len(p) + 1 for p in overlap_parts)
                else:
                    result.append(self._build_chunk(chunk, [sentence]))
                    i += 1

        if current_parts:
            result.append(self._build_chunk(chunk, current_parts))

        return result

    def _split_sentences(self, text: str) -> List[str]:
        try:
            import nltk
            sentences = nltk.sent_tokenize(text)
            return [s for s in sentences if s.strip()]
        except Exception:
            return self._simple_split(text)

    def _simple_split(self, text: str) -> List[str]:
        sentences = []
        for part in text.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").split("\n"):
            part = part.strip()
            if part:
                sentences.append(part)
        return sentences if sentences else [text]

    def _get_overlap(self, parts: List[str]) -> List[str]:
        if not parts:
            return []
        overlap_parts = []
        overlap_length = 0
        for p in reversed(parts):
            if overlap_length + len(p) + 1 <= self.chunk_overlap:
                overlap_parts.insert(0, p)
                overlap_length += len(p) + 1
            else:
                break
        return overlap_parts

    def _build_chunk(self, original: DocumentChunk, parts: List[str]) -> DocumentChunk:
        content = " ".join(parts)
        metadata = dict(original.metadata)
        metadata["char_length"] = len(content)
        return DocumentChunk(
            content=content,
            source_file=original.source_file,
            page=original.page,
            metadata=metadata
        )
