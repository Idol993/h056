from typing import List

from docurag.config import PROMPT_TEMPLATE
from docurag.retrieval.vector_store import RetrievedChunk


class PromptBuilder:
    def __init__(self, template: str = PROMPT_TEMPLATE):
        self.template = template

    def build(self, question: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        retrieved_docs = self._format_retrieved_docs(retrieved_chunks)
        return self.template.format(
            retrieved_docs=retrieved_docs,
            question=question
        )

    @staticmethod
    def _format_retrieved_docs(chunks: List[RetrievedChunk]) -> str:
        parts = []
        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.source_file
            if chunk.page:
                source += f" (第 {chunk.page} 页)"
            parts.append(f"[{idx}] 来源: {source}\n{chunk.content}")
        return "\n\n".join(parts)
