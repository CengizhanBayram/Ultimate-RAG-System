import logging
import threading
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class DataFileWatcher:
    """
    Watchdog-based file watcher that triggers index rebuild on data file changes.

    Used in --watch mode (interactive + auto-rebuild). A threading lock prevents
    concurrent rebuilds if multiple files are modified in rapid succession.
    """

    def __init__(self, pipeline, data_dir: Path):
        self.pipeline = pipeline
        self.data_dir = data_dir
        self.observer = Observer()
        self._rebuild_lock = threading.Lock()

    def start(self) -> None:
        handler = _RebuildHandler(self.pipeline, self._rebuild_lock)
        self.observer.schedule(handler, str(self.data_dir), recursive=False)
        self.observer.start()
        logger.info(f"Watching {self.data_dir} for file changes...")

    def stop(self) -> None:
        self.observer.stop()
        self.observer.join()
        logger.info("File watcher stopped.")


class _RebuildHandler(FileSystemEventHandler):
    WATCHED = {".txt", ".csv", ".json"}

    def __init__(self, pipeline, lock: threading.Lock):
        self.pipeline = pipeline
        self.lock = lock

    def on_modified(self, event):
        if Path(event.src_path).suffix in self.WATCHED:
            if self.lock.acquire(blocking=False):
                try:
                    logger.warning(
                        f"File changed: {event.src_path} — rebuilding index..."
                    )
                    self.pipeline.cache.invalidate_all()
                    self.pipeline.build_index(force_rebuild=True)
                    logger.info("Auto-rebuild complete.")
                except Exception as e:
                    logger.error(f"Auto-rebuild failed: {e}")
                finally:
                    self.lock.release()
