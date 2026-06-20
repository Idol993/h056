import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import click

from docurag.config import (
    LOG_BACKUP_COUNT,
    LOG_FILE,
    LOG_MAX_BYTES,
    UPLOAD_DIR,
)


def setup_logging(verbose: bool = False):
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = logging.DEBUG if verbose else logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)


def print_error(msg: str):
    click.echo(click.style(f"[错误] {msg}", fg="red"), err=True)


def print_success(msg: str):
    click.echo(click.style(msg, fg="green"))


def print_answer(msg: str):
    click.echo(click.style(msg, fg="white"))


def print_source(file: str, page: Optional[int], snippet: str):
    location = f"{file}"
    if page:
        location += f":{page}"
    click.echo(click.style(f"  - [{location}] {snippet[:100]}...", fg="#808080"))


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="启用详细日志")
@click.pass_context
def cli(ctx: click.Context, verbose: bool):
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.option("--path", "-p", type=click.Path(exists=True), default=None,
              help=f"文件或目录路径，默认: {UPLOAD_DIR}")
@click.option("--clear", "-c", is_flag=True, help="摄入前清空向量库")
def ingest(path: Optional[str], clear: bool):
    from docurag.ingestion import DocumentLoader, Embedder, TextSplitter
    from docurag.retrieval import VectorStore

    try:
        loader = DocumentLoader()
        splitter = TextSplitter()
        embedder = Embedder()
        vector_store = VectorStore()

        if clear:
            vector_store.clear()
            print_success("向量库已清空")

        target_path = Path(path) if path else UPLOAD_DIR
        click.echo(f"正在加载文档: {target_path}")

        if target_path.is_file():
            raw_chunks = loader.load_file(target_path)
        elif target_path.is_dir():
            raw_chunks = loader.load_directory(target_path)
        else:
            print_error(f"路径不存在: {target_path}")
            sys.exit(1)

        if not raw_chunks:
            print_error("未加载到任何文档内容")
            sys.exit(1)

        click.echo(f"加载了 {len(raw_chunks)} 个文档块，正在切分...")
        split_chunks = splitter.split(raw_chunks)
        click.echo(f"切分为 {len(split_chunks)} 个文本片段，正在向量化...")

        embeddings = embedder.embed_chunks(split_chunks)
        click.echo(f"向量化完成，正在存入向量库...")

        vector_store.add_documents(split_chunks, embeddings)
        total = vector_store.count()
        print_success(f"文档摄入完成！向量库共 {total} 条记录")

    except Exception as e:
        logging.exception("文档摄入失败")
        print_error(str(e))
        sys.exit(1)


@cli.command()
@click.argument("question")
@click.option("--no-stream", is_flag=True, help="禁用流式输出")
@click.option("--filter", "-f", "filter_file", default=None, help="按文件名过滤")
def query(question: str, no_stream: bool, filter_file: Optional[str]):
    from docurag.generation import LLMClient, PromptBuilder
    from docurag.ingestion import Embedder
    from docurag.retrieval import Retriever, Reranker, VectorStore

    try:
        vector_store = VectorStore()
        embedder = Embedder()
        reranker = Reranker()
        retriever = Retriever(vector_store, embedder, reranker)
        prompt_builder = PromptBuilder()
        llm_client = LLMClient()

        if vector_store.count() == 0:
            print_error("向量库为空，请先运行 'docurag ingest' 摄入文档")
            sys.exit(1)

        click.echo(click.style("正在检索相关文档...", fg="cyan"))
        retrieved = retriever.retrieve(question, filter_file=filter_file)

        if not retrieved:
            print_answer("未找到相关信息")
            return

        sources = Retriever.format_sources(retrieved)
        prompt = prompt_builder.build(question, retrieved)

        click.echo(click.style("\n答案:", fg="cyan", bold=True))

        if no_stream:
            answer = llm_client.generate(prompt, stream=False)
            print_answer(answer)
        else:
            try:
                from rich.console import Console
                from rich.live import Live
                from rich.text import Text

                console = Console()
                accumulated = ""
                text = Text(accumulated)

                with Live(text, console=console, refresh_per_second=10) as live:
                    stream = llm_client.generate(prompt, stream=True)
                    if stream:
                        for token in stream:
                            accumulated += token
                            text = Text(accumulated)
                            live.update(text)
                click.echo()
            except ImportError:
                answer = llm_client.generate(prompt, stream=False)
                print_answer(answer)

        click.echo(click.style("\n引用来源:", fg="cyan", bold=True))
        for src in sources:
            print_source(src["file"], src["page"], src["snippet"])

    except Exception as e:
        logging.exception("查询失败")
        print_error(str(e))
        sys.exit(1)


@cli.command()
@click.option("--host", "-h", default="127.0.0.1", help="监听地址")
@click.option("--port", "-p", default=8000, type=int, help="监听端口")
def serve(host: str, port: int):
    import uvicorn
    click.echo(f"启动 DocuRAG API 服务: http://{host}:{port}")
    click.echo(f"  POST /query    - 问答接口")
    click.echo(f"  GET  /ingest   - 触发文档摄入")
    click.echo(f"  GET  /status   - 查看状态")
    uvicorn.run("docurag.api:app", host=host, port=port, reload=False)


@cli.command(name="list")
def list_files():
    from docurag.retrieval import VectorStore

    try:
        vector_store = VectorStore()
        files = vector_store.list_files()
        total = vector_store.count()

        click.echo(f"向量库共 {total} 条记录，来自 {len(files)} 个文件:")
        for f in files:
            click.echo(f"  - {f}")
    except Exception as e:
        logging.exception("获取文件列表失败")
        print_error(str(e))
        sys.exit(1)


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
