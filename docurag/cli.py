import json
import logging
import sys
from datetime import datetime
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
        encoding="utf-8",
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


def print_warn(msg: str):
    try:
        click.echo(click.style(msg, fg="yellow"))
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


def _format_time(ts: float) -> str:
    if not ts:
        return "-"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def _parse_time_str(s: str) -> float:
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        raise click.BadParameter(f"无法解析时间: {s}")


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
@click.option("--force", "-f", is_flag=True, help="强制重新处理所有文件（忽略哈希校验）")
def ingest(path: Optional[str], clear: bool, force: bool):
    from docurag.ingestion import IngestionPipeline

    try:
        pipeline = IngestionPipeline()

        def _progress(msg: str):
            click.echo(f"  {msg}")

        target_path = Path(path) if path else UPLOAD_DIR

        if target_path.is_file():
            click.echo(f"正在摄入文件: {target_path}")
            result = pipeline.ingest_file(target_path, force=force, progress_cb=_progress)
            if result.success:
                print_success(f"{result.message}")
                if result.details:
                    d = result.details[0]
                    if d.status == "replaced":
                        click.echo(f"  片段变化: {d.removed_chunks} -> {d.replaced_chunks} (delta {d.chunk_delta:+d})")
                    if d.old_hash and d.new_hash and d.old_hash != d.new_hash:
                        click.echo(f"  哈希变化: {d.old_hash[:16]} -> {d.new_hash[:16]}")
            else:
                print_error(result.message)
                sys.exit(1)
        elif target_path.is_dir():
            click.echo(f"正在同步目录: {target_path} {'(强制重建)' if force else ''}")
            result = pipeline.sync_directory(target_path, clear_first=clear, force=force, progress_cb=_progress)
            if result.success:
                print_success(result.message)
                if result.removed_files:
                    click.echo(f"  已清理 ({len(result.removed_files)}): {', '.join(result.removed_files)}")
                if result.processed_files:
                    click.echo(f"  新增文件 ({len(result.processed_files)}): {', '.join(result.processed_files)}")
                if result.updated_files:
                    click.echo(f"  更新文件 ({len(result.updated_files)}): {', '.join(result.updated_files)}")
                if result.skipped_files:
                    click.echo(f"  未变化跳过 ({len(result.skipped_files)}): {', '.join(result.skipped_files)}")
                if result.failed_files:
                    print_warn(f"  失败文件 ({len(result.failed_files)}): {', '.join(result.failed_files)}")
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
@click.option("--filter", "-f", "filter_file", default=None, help="按文件名过滤（精确匹配）")
@click.option("--ext", "filter_ext", default=None, help="按扩展名过滤，如 .pdf / .docx")
@click.option("--after", "updated_after", default=None, help="只查此时间后更新的文档 (YYYY-MM-DD 或时间戳)")
@click.option("--before", "updated_before", default=None, help="只查此时间前更新的文档")
@click.option("--retrieve-only", is_flag=True, help="仅返回检索到的来源片段，不调用 LLM")
@click.option("--top-k", default=None, type=int, help="返回前 K 条结果")
@click.option("--debug", is_flag=True, help="调试模式：显示过滤条件、命中文件、相似度/重排序分数")
@click.option("--json", "json_output", is_flag=True, help="以 JSON 格式输出完整结果（含调试信息）")
def query(
    question: str,
    no_stream: bool,
    filter_file: Optional[str],
    filter_ext: Optional[str],
    updated_after: Optional[str],
    updated_before: Optional[str],
    retrieve_only: bool,
    top_k: Optional[int],
    debug: bool,
    json_output: bool,
):
    from docurag.generation import LLMClient, PromptBuilder
    from docurag.ingestion import Embedder
    from docurag.retrieval import (
        RetrieveFilters,
        Retriever,
        Reranker,
        VectorStore,
    )

    try:
        filters = RetrieveFilters(
            file_name=filter_file,
            file_ext=filter_ext,
            updated_after=_parse_time_str(updated_after) if updated_after else None,
            updated_before=_parse_time_str(updated_before) if updated_before else None,
        )

        vector_store = VectorStore()
        embedder = Embedder()
        reranker = Reranker()
        retriever = Retriever(
            vector_store,
            embedder,
            reranker,
            **({"top_k": top_k, "rerank_top_k": min(top_k, 5)} if top_k else {}),
        )

        if vector_store.count() == 0:
            print_error("向量库为空，请先运行 'docurag ingest' 摄入文档")
            sys.exit(1)

        if not json_output:
            click.echo(click.style("正在检索相关文档...", fg="cyan"))
        retrieved, debug_info = retriever.retrieve(question, filters=filters, return_debug=True)
        debug_dict = Retriever.debug_to_dict(debug_info)

        if not retrieved:
            if json_output:
                click.echo(json.dumps({
                    "question": question,
                    "answer": "未找到相关信息",
                    "sources": [],
                    "debug": debug_dict,
                }, ensure_ascii=False, indent=2))
            else:
                print_answer("未找到相关信息")
                if debug:
                    click.echo(click.style("\n调试信息:", fg="yellow", bold=True))
                    click.echo(json.dumps(debug_dict, ensure_ascii=False, indent=2))
            return

        sources = Retriever.format_sources(retrieved)

        if json_output:
            output = {
                "question": question,
                "sources": sources,
                "debug": debug_dict,
            }

        if not json_output and (debug or not retrieve_only):
            click.echo(click.style(f"\n命中 {len(retrieved)} 个相关片段:", fg="cyan", bold=True))
            for i, src in enumerate(sources, 1):
                print_source(src.get("file", "?"), src.get("page"), src.get("snippet", ""))
                if debug:
                    score_info = []
                    if src.get("vector_score") is not None:
                        score_info.append(f"vec={src['vector_score']:.4f}")
                    if src.get("rerank_score") is not None:
                        score_info.append(f"rerank={src['rerank_score']:.4f}")
                    if score_info:
                        click.echo(f"      ({', '.join(score_info)})")

        if debug and not json_output:
            click.echo(click.style("\n调试信息:", fg="yellow", bold=True))
            click.echo(json.dumps(debug_dict, ensure_ascii=False, indent=2))

        if retrieve_only:
            if json_output:
                output["answer"] = ""
                click.echo(json.dumps(output, ensure_ascii=False, indent=2))
            return

        prompt_builder = PromptBuilder()
        llm_client = LLMClient()
        prompt = prompt_builder.build(question, retrieved)

        if not json_output:
            click.echo(click.style("\n答案:", fg="cyan", bold=True))

        if no_stream:
            try:
                answer = llm_client.generate(prompt, stream=False) or "未找到相关信息"
                if not json_output:
                    print_answer(answer)
            except Exception as e:
                logging.exception("LLM 调用失败")
                answer = ""
                if not json_output:
                    print_error(f"生成答案失败: {e}")
        else:
            answer = ""
            try:
                from rich.console import Console
                from rich.live import Live
                from rich.text import Text

                if json_output:
                    answer = llm_client.generate(prompt, stream=False) or "未找到相关信息"
                else:
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
                    answer = llm_client.generate(prompt, stream=False) or "未找到相关信息"
                    if not json_output:
                        print_answer(answer)
                except Exception as e:
                    logging.exception("LLM 调用失败")
                    answer = ""
                    if not json_output:
                        print_error(f"生成答案失败: {e}")

        if json_output:
            output["answer"] = answer
            click.echo(json.dumps(output, ensure_ascii=False, indent=2))
            return

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
@click.option("--no-auto-ingest", is_flag=True, help="启动时不自动摄入已有文件")
def serve(host: str, port: int, no_watch: bool, no_auto_ingest: bool):
    import uvicorn
    from docurag.api import set_auto_watch, set_auto_ingest_on_start

    set_auto_watch(not no_watch)
    set_auto_ingest_on_start(not no_auto_ingest)

    click.echo(f"启动 DocuRAG API 服务: http://{host}:{port}")
    click.echo(f"  POST /query                 - 问答接口 (支持 debug/filter_ext/retrieve_only)")
    click.echo(f"  GET  /ingest                - 触发文档同步")
    click.echo(f"  GET  /status                - 查看状态")
    click.echo(f"  GET  /doctor                - 审计诊断")
    click.echo(f"  POST /doctor/fix            - 一键修复 (返回详细分类报告)")
    click.echo(f"  GET  /docs                  - 文件列表 (含失败文件)")
    click.echo(f"  GET  /docs/{{name}}          - 文件详情")
    click.echo(f"  GET  /docs/{{name}}/history  - 文件摄入历史")
    click.echo(f"  POST /docs/{{name}}/ingest   - 重新摄入单文件")
    click.echo(f"  DELETE /docs/{{name}}        - 删除单文件记录")
    if not no_watch:
        click.echo(f"  自动监听: {UPLOAD_DIR} (新增/修改/删除/重命名自动同步)")
    if not no_auto_ingest and not no_watch:
        click.echo(f"  启动自动同步: uploads 已有文件自动入库")

    uvicorn.run("docurag.api:app", host=host, port=port, reload=False)


@cli.command(name="list")
@click.option("--short", "-s", is_flag=True, help="简洁显示，仅文件名")
def list_files(short: bool):
    from docurag.ingestion import IngestionPipeline

    try:
        pipeline = IngestionPipeline()
        files = pipeline.list_files(upload_dir=UPLOAD_DIR)
        total = pipeline.vector_store.count()

        if short:
            click.echo(f"共 {total} 条记录，{len(files)} 个文件:")
            for f in files:
                status = "✓" if f.exists_in_uploads else "✗"
                click.echo(f"  [{status}] {f.filename}")
            return

        click.echo(f"向量库共 {total} 条记录，来自 {len(files)} 个文件:")
        if not files:
            click.echo("  (空库)")
            return

        source_map = {"manual": "手动", "watcher": "监听器", "api": "Web API", "doctor": "诊断修复", "": "-"}

        for f in files:
            status = "在库中" if f.exists_in_uploads else "[已删除]"
            status_color = "green" if f.exists_in_uploads else "red"
            hash_short = f.file_hash[:16] if f.file_hash else "-"
            last_status = f.last_ingest_status or "unknown"
            status_map = {"added": "新增", "replaced": "更新", "skipped": "跳过", "failed": "失败", "": "-"}
            status_label = status_map.get(last_status, last_status)
            state = pipeline.vector_store.get_file_ingest_state(f.filename) or {}
            last_source = source_map.get(state.get("source", ""), state.get("source", "-"))
            try:
                status_str = click.style(status, fg=status_color)
            except Exception:
                status_str = status

            click.echo(f"  {f.filename}")
            click.echo(
                f"    片段: {f.chunk_count}  |  更新: {_format_time(f.updated_at)}  |  {status_str}"
                f"  |  上次: {status_label}  |  来源: {last_source}"
            )
            if f.last_ingest_error:
                click.echo(f"    错误: {f.last_ingest_error}")
            if f.prev_chunk_count and f.prev_chunk_count != f.chunk_count:
                delta = f.chunk_count - f.prev_chunk_count
                click.echo(f"    上次片段数: {f.prev_chunk_count} (delta {delta:+d})")
            click.echo(f"    哈希: {hash_short}")
    except Exception as e:
        logging.exception("获取文件列表失败")
        print_error(str(e))
        sys.exit(1)


@cli.command()
@click.argument("filename")
@click.option("--yes", "-y", is_flag=True, help="跳过确认")
def remove(filename: str, yes: bool):
    from docurag.ingestion import IngestionPipeline

    try:
        pipeline = IngestionPipeline()
        info = pipeline.get_file_info(filename)

        if not info:
            print_error(f"向量库中没有文件: {filename}")
            sys.exit(1)

        if not yes:
            click.echo(f"将从向量库删除: {filename} ({info.chunk_count} 个片段)")
            confirm = click.prompt("确认删除? (y/N)", default="N", show_default=False)
            if confirm.lower() not in ("y", "yes"):
                click.echo("已取消")
                return

        if pipeline.remove_file(filename):
            print_success(f"已删除文件: {filename}")
        else:
            print_error(f"删除失败: {filename}")
            sys.exit(1)

    except Exception as e:
        logging.exception("删除文件失败")
        print_error(str(e))
        sys.exit(1)


@cli.command(name="sync")
@click.option("--clear", "-c", is_flag=True, help="同步前清空向量库")
@click.option("--force", "-f", is_flag=True, help="强制重建所有文件")
def sync_cmd(clear: bool, force: bool):
    from docurag.ingestion import IngestionPipeline

    try:
        pipeline = IngestionPipeline()

        def _progress(msg: str):
            click.echo(f"  {msg}")

        click.echo(f"正在同步 uploads 目录: {UPLOAD_DIR} {'(强制重建)' if force else ''}")
        result = pipeline.sync_directory(UPLOAD_DIR, clear_first=clear, force=force, progress_cb=_progress)

        if result.success:
            print_success(result.message)
            if result.removed_files:
                click.echo(f"  已清理: {', '.join(result.removed_files)}")
            if result.processed_files:
                click.echo(f"  新增: {', '.join(result.processed_files)}")
            if result.updated_files:
                click.echo(f"  更新: {', '.join(result.updated_files)}")
            if result.skipped_files:
                click.echo(f"  跳过: {', '.join(result.skipped_files)}")
        else:
            print_error(result.message)
            sys.exit(1)

    except Exception as e:
        logging.exception("同步失败")
        print_error(str(e))
        sys.exit(1)


@cli.command()
@click.option("--fix", is_flag=True, help="发现问题后自动修复")
@click.option("--yes", "-y", is_flag=True, help="修复时跳过确认")
def doctor(fix: bool, yes: bool):
    from docurag.ingestion import IngestionPipeline

    try:
        pipeline = IngestionPipeline()
        report = pipeline.doctor(upload_dir=UPLOAD_DIR, watcher_running=False)

        click.echo(click.style("DocuRAG 文档库诊断报告", fg="cyan", bold=True))
        click.echo(f"  Uploads 目录: {report.upload_dir}")
        click.echo(f"  向量库目录:  {report.db_dir}")
        click.echo(f"  监听器状态:  {'运行中' if report.watcher_running else '未运行'}")
        click.echo()
        click.echo(f"  Uploads 支持文件数: {report.total_in_uploads}")
        click.echo(f"  向量库文件数:      {report.total_in_db}")
        click.echo()

        issues = []

        def _section(title: str, items: list, level: str = "warn"):
            color = {"ok": "green", "warn": "yellow", "err": "red"}.get(level, "yellow")
            if items:
                issues.append((title, items))
                try:
                    click.echo(click.style(f"  [{title}] {len(items)} 项:", fg=color, bold=True))
                except Exception:
                    click.echo(f"  [{title}] {len(items)} 项:")
                for it in items:
                    click.echo(f"    - {it}")
                click.echo()
            else:
                try:
                    click.echo(click.style(f"  [{title}] 0 项 ✓", fg="green"))
                except Exception:
                    click.echo(f"  [{title}] 0 项 ✓")

        _section("孤儿记录(库有目录无)", report.orphan_files, level="err")
        _section("缺失文件(目录有库无)", report.missing_files, level="warn")
        _section("哈希不一致(可能过期)", report.hash_mismatch, level="warn")
        _section("文件较目录旧", report.stale_files, level="warn")
        _section("空文件", report.empty_files, level="warn")
        _section("不支持格式", report.unsupported_files, level="warn")

        if not issues:
            print_success("\n所有检查通过，文档库状态良好。")
            return

        if not fix:
            print_warn("\n运行 'docurag doctor --fix' 自动修复上述问题。")
            return

        if not yes:
            click.echo()
            confirm = click.prompt("是否自动修复上述问题? (y/N)", default="N", show_default=False)
            if confirm.lower() not in ("y", "yes"):
                click.echo("已取消")
                return

        def _progress(msg: str):
            click.echo(f"  {msg}")

        fix = pipeline.fix_doctor_issues(report, progress_cb=_progress)
        print_success(f"\n{fix.message}")

        if fix.cleaned_orphans:
            click.echo(f"  已清理孤儿 ({len(fix.cleaned_orphans)}): {', '.join(fix.cleaned_orphans)}")
        if fix.added_missing:
            click.echo(f"  已补录缺失 ({len(fix.added_missing)}): {', '.join(fix.added_missing)}")
        if fix.fixed_hash_mismatch:
            click.echo(f"  已修复哈希不一致 ({len(fix.fixed_hash_mismatch)}): {', '.join(fix.fixed_hash_mismatch)}")
        if fix.rebuilt_stale:
            click.echo(f"  已重建过期 ({len(fix.rebuilt_stale)}): {', '.join(fix.rebuilt_stale)}")

        all_failed = fix.rebuilt_stale_failed + fix.added_missing_failed + fix.fixed_hash_mismatch_failed
        if all_failed:
            print_warn(f"  仍失败 ({len(all_failed)}): {', '.join(all_failed)}")
            for d in fix.details:
                if d.status == "failed" and d.error:
                    click.echo(f"    - {d.filename}: {d.error}")

    except Exception as e:
        logging.exception("诊断失败")
        print_error(str(e))
        sys.exit(1)


@cli.command()
@click.argument("filename")
@click.option("--limit", "-n", default=10, type=int, help="显示最近 N 条记录 (默认 10)")
def history(filename: str, limit: int):
    """查看文件的摄入历史记录"""
    from docurag.ingestion import IngestionPipeline

    try:
        pipeline = IngestionPipeline()
        entries = pipeline.get_ingest_history(filename, limit=limit)
        if not entries:
            print_warn(f"文件 {filename} 暂无摄入历史记录")
            return

        status_map = {"added": "新增", "replaced": "更新", "skipped": "跳过", "failed": "失败"}
        source_map = {"manual": "手动", "watcher": "监听器", "api": "Web API", "doctor": "诊断修复"}

        click.echo(f"文件 {filename} 的最近 {len(entries)} 次摄入记录:")
        for i, e in enumerate(entries, 1):
            status_label = status_map.get(e.status, e.status)
            source_label = source_map.get(e.source, e.source)
            ts_str = _format_time(e.timestamp)
            delta_str = f"{e.chunk_delta:+d}" if e.chunk_delta != 0 else "0"
            line = f"  #{i} [{ts_str}] {status_label} | 来源: {source_label} | 片段: {e.chunk_count} (Δ {delta_str})"
            if e.status == "failed":
                print_warn(line)
                if e.error:
                    click.echo(f"       错误: {e.error}")
            else:
                click.echo(line)
                if e.prev_file_hash and e.file_hash and e.prev_file_hash != e.file_hash:
                    click.echo(f"       哈希: {e.prev_file_hash[:12]} → {e.file_hash[:12]}")
    except Exception as e:
        logging.exception("获取历史失败")
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
