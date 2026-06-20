import logging
import threading
import time
from pathlib import Path
from typing import Optional

from docurag.config import UPLOAD_DIR
from docurag.ingestion.loader import DocumentLoader
from docurag.ingestion.pipeline import IngestionPipeline
from docurag.retrieval.vector_store import compute_file_hash

logger = logging.getLogger(__name__)


class DirectoryWatcher:
    def __init__(
        self,
        watch_dir: str | Path = UPLOAD_DIR,
        pipeline: Optional[IngestionPipeline] = None,
        debounce_seconds: float = 2.0
    ):
        self.watch_dir = Path(watch_dir)
        self.pipeline = pipeline or IngestionPipeline()
        self.debounce_seconds = debounce_seconds
        self._supported = DocumentLoader.SUPPORTED_EXTENSIONS
        self._observer = None
        self._running = False
        self._pending_files: dict[str, float] = {}
        self._processed_hashes: dict[str, str] = {}
        self._lock = threading.Lock()
        self._debounce_thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return

        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self._running = True

        self._init_existing_hashes()

        try:
            from watchdog.events import FileSystemEventHandler, FileMovedEvent
            from watchdog.observers import Observer

            watcher = self

            class _Handler(FileSystemEventHandler):
                def on_created(self, event):
                    if event.is_directory:
                        return
                    watcher._enqueue(event.src_path)

                def on_modified(self, event):
                    if event.is_directory:
                        return
                    watcher._enqueue(event.src_path)

                def on_moved(self, event):
                    if event.is_directory:
                        return
                    if isinstance(event, FileMovedEvent):
                        src_path = Path(event.src_path)
                        if src_path.suffix.lower() in watcher._supported:
                            watcher._on_renamed(event.src_path, event.dest_path)
                        watcher._enqueue(event.dest_path)

                def on_deleted(self, event):
                    if event.is_directory:
                        return
                    watcher._on_deleted(event.src_path)

            self._observer = Observer()
            self._observer.schedule(_Handler(), str(self.watch_dir), recursive=False)
            self._observer.start()

            self._debounce_thread = threading.Thread(target=self._debounce_loop, daemon=True)
            self._debounce_thread.start()

            logger.info(f"目录监听器已启动: {self.watch_dir}")
            self._scan_existing()

        except ImportError:
            logger.warning("watchdog 未安装，自动监听功能不可用。请运行: pip install watchdog")
            self._scan_existing()

    def stop(self):
        self._running = False
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception:
                pass
            self._observer = None
        logger.info("目录监听器已停止")

    def _init_existing_hashes(self):
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(self.watch_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in self._supported:
                try:
                    h = compute_file_hash(f)
                    if h:
                        self._processed_hashes[str(f)] = h
                except Exception:
                    pass

    def _scan_existing(self):
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(self.watch_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in self._supported:
                self._enqueue(str(f), check_hash=False)

    def _enqueue(self, file_path: str, check_hash: bool = True):
        path = Path(file_path)
        if path.suffix.lower() not in self._supported:
            return
        with self._lock:
            self._pending_files[str(path)] = time.time()
        logger.debug(f"文件变更已排队: {path.name} (check_hash={check_hash})")

    def _on_deleted(self, file_path: str):
        path = Path(file_path)
        if path.suffix.lower() not in self._supported:
            return
        try:
            self.pipeline.remove_file(path.name)
            with self._lock:
                self._processed_hashes.pop(str(path), None)
        except Exception as e:
            logger.error(f"处理删除事件失败 {path.name}: {e}")

    def _on_renamed(self, src_path: str, dest_path: str):
        src = Path(src_path)
        dest = Path(dest_path)
        try:
            self.pipeline.remove_file(src.name)
            with self._lock:
                self._processed_hashes.pop(str(src), None)
            logger.info(f"检测到重命名: {src.name} -> {dest.name}，已移除旧记录")
        except Exception as e:
            logger.error(f"处理重命名事件失败: {e}")

    def _debounce_loop(self):
        while self._running:
            try:
                time.sleep(0.5)
                now = time.time()
                ready: list[str] = []

                with self._lock:
                    for fp, ts in list(self._pending_files.items()):
                        if now - ts >= self.debounce_seconds:
                            try:
                                p = Path(fp)
                                if not p.exists() or p.stat().st_size == 0:
                                    self._pending_files.pop(fp, None)
                                    continue

                                current_hash = compute_file_hash(p)
                                prev_hash = self._processed_hashes.get(fp)

                                if current_hash != prev_hash:
                                    ready.append(fp)
                                    self._processed_hashes[fp] = current_hash
                                else:
                                    logger.debug(f"文件内容未变化，跳过: {p.name}")

                            except FileNotFoundError:
                                pass
                            except Exception as e:
                                logger.warning(f"检查文件哈希失败 {fp}: {e}")

                            self._pending_files.pop(fp, None)

                for fp in ready:
                    self._process_file(fp)

            except Exception as e:
                logger.exception(f"防抖循环异常: {e}")

    def _process_file(self, file_path: str):
        path = Path(file_path)
        if not path.exists():
            return
        logger.info(f"自动摄入: {path.name}")
        try:
            result = self.pipeline.ingest_file(path, force=False)
            if result.success:
                logger.info(f"自动摄入完成: {result.message}")
            else:
                logger.warning(f"自动摄入失败: {path.name} - {result.message}")
        except Exception as e:
            logger.exception(f"自动摄入异常 {path.name}: {e}")
