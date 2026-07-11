"""
Disk-backed persistence for the case file + compaction history.

Layout (mirrors reports/health/history.jsonl from the training health
monitor):

    <local_run_dir>/compaction/
        case_file.json     # authoritative structured case file
        case_file.md       # human-readable rendering, regenerated on save
        history.jsonl      # one CompactionEvent per line

Atomic writes use a temp file with pid+nanos suffix, fsync, os.replace,
plus an fcntl exclusive lock so a concurrent reader can't see a partial file.
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from capo.context.compaction.types import CaseFile, CompactionEvent

logger = logging.getLogger(__name__)


class CompactionStore:
    """Reads and writes the per-run compaction artifacts."""

    def __init__(self, local_run_dir: Path | str) -> None:
        self.dir = Path(local_run_dir).expanduser() / "compaction"

    # ------------------------------------------------------------------ #
    # Path helpers                                                        
    # ------------------------------------------------------------------ #

    @property
    def case_file_json(self) -> Path:
        return self.dir / "case_file.json"

    @property
    def case_file_md(self) -> Path:
        return self.dir / "case_file.md"

    @property
    def history(self) -> Path:
        return self.dir / "history.jsonl"

    @property
    def _lock_path(self) -> Path:
        return self.dir / ".lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    # ------------------------------------------------------------------ #
    # CRUD                                                                
    # ------------------------------------------------------------------ #

    def load_case_file(self) -> CaseFile | None:
        """Return the persisted case file, or None if missing/corrupt."""
        try:
            with self._locked():
                text = self.case_file_json.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning("Failed to read case file %s: %s", self.case_file_json, exc)
            return None
        try:
            return CaseFile.from_json(text)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Corrupt case file %s: %s", self.case_file_json, exc)
            return None

    def save_case_file(self, case_file: CaseFile) -> None:
        """Atomically write the case file (JSON + markdown rendering)."""
        with self._locked():
            self._atomic_write(self.case_file_json, case_file.to_json())
            self._atomic_write(self.case_file_md, case_file.to_markdown())

    def append_event(self, event: CompactionEvent) -> None:
        """Append one event row to history.jsonl."""
        with self._locked():
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.history, "a", encoding="utf-8") as f:
                f.write(event.to_json_line() + "\n")

    # ------------------------------------------------------------------ #
    # Internals                                                           
    # ------------------------------------------------------------------ #

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}.{time.time_ns()}")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                os.write(fd, text.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, path)
        except OSError:
            try:
                tmp.unlink()
            except OSError as cleanup_exc:
                if cleanup_exc.errno != errno.ENOENT:
                    logger.debug("tmp cleanup failed for %s: %s", tmp, cleanup_exc)
            raise
