"""Thread-safe JSONL recording for labeled Rokoko motion sessions."""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4


class SessionRecorder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._lock = threading.Lock()
        self._file: TextIO | None = None
        self._session: dict[str, Any] | None = None
        self._frame_count = 0
        self._error: str | None = None
        self._last_file: str | None = None

    def start(self, label: object, device_id: str | int | None) -> dict[str, Any]:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("A non-empty action label is required")
        label = label.strip()
        if len(label) > 120:
            raise ValueError("Action labels must be 120 characters or fewer")

        with self._lock:
            if self._file is not None:
                raise RuntimeError("A recording is already active")

            started_at = datetime.now(timezone.utc)
            session_id = uuid4().hex
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"{started_at:%Y%m%dT%H%M%SZ}_{session_id[:8]}.jsonl"
            stream = path.open("x", encoding="utf-8")
            session = {
                "record_type": "session",
                "session_id": session_id,
                "label": label,
                "device_id": device_id,
                "started_at": started_at.isoformat(),
            }
            try:
                stream.write(json.dumps(session, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
            except (OSError, TypeError, ValueError):
                stream.close()
                raise

            self._file = stream
            self._session = session
            self._frame_count = 0
            self._error = None
            self._last_file = str(path)
            return self._snapshot_locked()

    def record_group(
        self,
        device_id: str | int | None,
        group_name: str,
        timestamp_us: int,
        streams: dict[str, dict[str, list[int | float]]],
    ) -> None:
        with self._lock:
            if self._file is None or self._session is None:
                return

            frame = {
                "record_type": "frame",
                "session_id": self._session["session_id"],
                "label": self._session["label"],
                "device_id": device_id,
                "group_name": group_name,
                "timestamp_us": timestamp_us,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "streams": streams,
            }
            try:
                self._file.write(json.dumps(frame, separators=(",", ":"), allow_nan=False) + "\n")
                self._frame_count += 1
                if self._frame_count % 32 == 0:
                    self._file.flush()
            except (OSError, TypeError, ValueError) as exc:
                self._error = f"Recording stopped: {exc}"
                print(f"# {self._error}", file=sys.stderr, flush=True)
                self._close_locked()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._file is None:
                raise RuntimeError("No recording is active")
            self._close_locked()
            return self._snapshot_locked()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._close_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "active": self._file is not None,
            "session_id": self._session.get("session_id") if self._session else None,
            "label": self._session.get("label") if self._session else None,
            "frames": self._frame_count,
            "file": self._last_file,
            "error": self._error,
        }

    def _close_locked(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
            except OSError as exc:
                self._error = f"Could not flush recording: {exc}"
                print(f"# {self._error}", file=sys.stderr, flush=True)
            finally:
                self._file.close()
                self._file = None
