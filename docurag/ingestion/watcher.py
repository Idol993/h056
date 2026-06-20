import logging
import threading
import time
from pathlib import Path
from typing import Optional, Set

from docurag.config import UPLOAD_DIR
from docurag.ingestion.loader import DocumentLoader
from docurag.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


class DirectoryWatcher:
    def __init__(
        self,
        watch_dir: str | Path = UPLOAD_DIR,
        pipeline: Optional[IngestionPipeline] = None,
        debounce_seconds: float = 1.5
    ):
        self.watch_dir = Path(watch_dir)
        self.pipeline = pipeline or IngestionPipeline()
        self.debounce_seconds = debounce_seconds
        self._supported = DocumentLoader.SUPPORTED_EXTENSIONS
        self._observer = None
        self._running = False
        self._pending_files: dict[str, float] = {}
        self._processed_sizes: dict[str, int] = {}
        self._lock = threading.Lock()
        self._debounce_thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return

        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self._running = True

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

    def _scan_existing(self):
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(self.watch_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in self._supported:
                self._enqueue(str(f))

    def _enqueue(self, file_path: str):
        path = Path(file_path)
        if path.suffix.lower() not in self._supported:
            return
        with self._lock:
            self._pending_files[str(path)] = time.time()
        logger.debug(f"文件变更已排队: {path.name}")

    def _on_deleted(self, file_path: str):
        path = Path(file_path)
        if path.suffix.lower() not in self._supported:
            return
        try:
            self.pipeline.remove_file(path.name)
            self._processed_sizes.pop(str(path), None)
        except Exception as e:
            logger.error(f"处理删除事件失败 {path.name}: {e}")

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
                                size = Path(fp).stat().st_size
                            except FileNotFoundError:
                                self._pending_files.pop(fp, None)
                                continue
                            prev_size = self._processed_sizes.get(fp)
                            if size > 0 and size != prev_size:
                                ready.append(fp)
                                self._processed_sizes[fp] = size
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
            result = self.pipeline.ingest_file(path)
            if result.success:
                logger.info(f"自动摄入完成: {result.message}")
            else:
                logger.warning(f"自动摄入失败: {path.name} - {result.message}")
        except Exception as e:
            logger.exception(f"自动摄入异常 {path.name}: {e}")
