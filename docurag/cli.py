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

    root_logger.handlers.clear()

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
    try:
        click.echo(click.style(f"[错误] {msg}", fg="red"), err=True)
    except Exception:
        click.echo(f"[错误] {msg}", err=True)


def print_success(msg: str):
    try:
        click.echo(click.style(msg, fg="green"))
    except Exception:
        click.echo(msg)


def print_answer(msg: str):
    try:
        click.echo(click.style(msg, fg="white"))
    except Exception:
        click.echo(msg)


def print_source(file: str, page: Optional[int], snippet: str):
    location = f"{file}"
    if page:
        location += f":{page}"
    safe_snippet = (snippet or "")[:100].replace("\n", " ")
    display = f"  - [{location}] {safe_snippet}..."
    try:
        click.echo(click.style(display, fg="bright_black"))
    except Exception:
        try:
            click.echo(click.style(display, dim=True))
        except Exception:
            click.echo(display)


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
    from docurag.ingestion import IngestionPipeline

    try:
        pipeline = IngestionPipeline()

        def _progress(msg: str):
            click.echo(f"  {msg}")

        target_path = Path(path) if path else UPLOAD_DIR

        if target_path.is_file():
            click.echo(f"正在摄入文件: {target_path}")
            result = pipeline.ingest_file(target_path, progress_cb=_progress)
            if result.success:
                print_success(f"{result.message}")
            else:
                print_error(result.message)
                sys.exit(1)
        elif target_path.is_dir():
            click.echo(f"正在摄入目录: {target_path}")
            result = pipeline.ingest_directory(target_path, clear_first=clear, progress_cb=_progress)
            if result.success:
                print_success(result.message)
                if result.processed_files:
                    click.echo(f"  已处理: {', '.join(result.processed_files)}")
                if result.skipped_files:
                    click.echo(f"  已跳过: {', '.join(result.skipped_files)}")
            else:
                print_error(result.message)
                sys.exit(1)
        else:
            print_error(f"路径不存在: {target_path}")
            sys.exit(1)

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
            try:
                answer = llm_client.generate(prompt, stream=False)
                print_answer(answer or "未找到相关信息")
            except Exception as e:
                logging.exception("LLM 调用失败")
                print_error(f"生成答案失败: {e}")
                answer = ""
        else:
            answer = ""
            try:
                from rich.console import Console
                from rich.live import Live
                from rich.text import Text

                console = Console()
                accumulated = ""
                text = Text(accumulated)

                with Live(text, console=console, refresh_per_second=10) as live:
                    try:
                        stream = llm_client.generate(prompt, stream=True)
                        if stream:
                            for token in stream:
                                accumulated += token
                                text = Text(accumulated)
                                live.update(text)
                    except Exception as inner_e:
                        accumulated += f"\n[生成中断: {inner_e}]"
                        text = Text(accumulated)
                        live.update(text)
                answer = accumulated
                click.echo()
            except ImportError:
                try:
                    answer = llm_client.generate(prompt, stream=False)
                    print_answer(answer or "未找到相关信息")
                except Exception as e:
                    logging.exception("LLM 调用失败")
                    print_error(f"生成答案失败: {e}")
                    answer = ""

        click.echo()
        try:
            click.echo(click.style("引用来源:", fg="cyan", bold=True))
        except Exception:
            click.echo("引用来源:")

        if sources:
            for src in sources:
                try:
                    print_source(src.get("file", "?"), src.get("page"), src.get("snippet", ""))
                except Exception as e:
                    logging.debug(f"打印来源失败: {e}")
                    location = src.get("file", "?")
                    if src.get("page"):
                        location += f":{src['page']}"
                    s = (src.get("snippet") or "")[:80]
                    click.echo(f"  - [{location}] {s}...")
        else:
            click.echo("  (无来源)")

    except Exception as e:
        logging.exception("查询失败")
        print_error(str(e))
        sys.exit(1)


@cli.command()
@click.option("--host", "-h", default="127.0.0.1", help="监听地址")
@click.option("--port", "-p", default=8000, type=int, help="监听端口")
@click.option("--no-watch", is_flag=True, help="禁用 uploads 目录自动监听")
def serve(host: str, port: int, no_watch: bool):
    import uvicorn
    from docurag.api import set_auto_watch

    set_auto_watch(not no_watch)

    click.echo(f"启动 DocuRAG API 服务: http://{host}:{port}")
    click.echo(f"  POST /query    - 问答接口")
    click.echo(f"  GET  /ingest   - 触发文档摄入")
    click.echo(f"  GET  /status   - 查看状态")
    if not no_watch:
        click.echo(f"  自动监听: {UPLOAD_DIR} (新增/修改文件自动摄入)")

    uvicorn.run("docurag.api:app", host=host, port=port, reload=False)


@cli.command(name="list")
def list_files():
    from docurag.retrieval import VectorStore

    try:
        vector_store = VectorStore()
        files = vector_store.list_files()
        total = vector_store.count()

        click.echo(f"向量库共 {total} 条记录，来自 {len(files)} 个文件:")
        if files:
            for f in files:
                click.echo(f"  - {f}")
        else:
            click.echo("  (空库)")
    except Exception as e:
        logging.exception("获取文件列表失败")
        print_error(str(e))
        sys.exit(1)


def main():
    try:
        cli(obj={})
    except KeyboardInterrupt:
        click.echo("\n已取消")
        sys.exit(0)


if __name__ == "__main__":
    main()
