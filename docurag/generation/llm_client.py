import logging
from typing import Iterator, Optional

from docurag.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url, timeout=self.timeout)
            except ImportError:
                raise ImportError("请安装 ollama: pip install ollama")
        return self._client

    def generate(self, prompt: str, stream: bool = False) -> Optional[str | Iterator[str]]:
        client = self._get_client()
        logger.info(f"调用 LLM 模型: {self.model} (stream={stream})")

        try:
            if stream:
                return self._generate_stream(client, prompt)
            else:
                response = client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False
                )
                return response["message"]["content"]
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

    def _generate_stream(self, client, prompt: str) -> Iterator[str]:
        stream = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

    def check_connection(self) -> bool:
        try:
            client = self._get_client()
            client.list()
            logger.info("Ollama 连接正常")
            return True
        except Exception as e:
            logger.warning(f"Ollama 连接失败: {e}")
            return False
