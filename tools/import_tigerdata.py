"""Import recorded motion JSONL sessions into a Tiger Data TimescaleDB database."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS glove_motion_frames (
    session_id TEXT NOT NULL,
    group_name TEXT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    device_id TEXT,
    timestamp_us BIGINT NOT NULL,
    label TEXT NOT NULL,
    streams JSONB NOT NULL,
    PRIMARY KEY (session_id, group_name, captured_at)
)
"""

INSERT_FRAME = """
INSERT INTO glove_motion_frames
    (session_id, group_name, captured_at, device_id, timestamp_us, label, streams)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (session_id, group_name, captured_at) DO NOTHING
"""


def read_frames(path: Path) -> Iterator[dict[str, Any]]:
    session: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each record must be an object")
            if record.get("record_type") == "session":
                session = record
            elif record.get("record_type") == "frame":
                if session is None:
                    raise ValueError(f"{path}:{line_number}: frame appears before session header")
                if not isinstance(record.get("streams"), dict):
                    raise ValueError(f"{path}:{line_number}: streams must be an object")
                try:
                    captured_at = datetime.fromisoformat(record["captured_at"])
                    if captured_at.tzinfo is None or captured_at.utcoffset() is None:
                        raise ValueError("captured_at must include a timezone")
                    timestamp_us = int(record["timestamp_us"])
                    session_id = str(record["session_id"])
                    group_name = str(record["group_name"])
                    label = str(record["label"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: missing or invalid frame fields") from exc
                yield {
                    "session_id": session_id,
                    "group_name": group_name,
                    "captured_at": captured_at,
                    "device_id": (
                        str(record["device_id"]) if record.get("device_id") is not None else None
                    ),
                    "timestamp_us": timestamp_us,
                    "label": label,
                    "streams": record["streams"],
                }


def import_files(database_url: str, paths: list[Path]) -> int:
    try:
        import psycopg
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        raise RuntimeError(
            "Tiger Data import requires psycopg; install it with "
            "`py -m pip install -r requirements-data.txt`."
        ) from exc

    inserted = 0
    try:
        with psycopg.connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
                cursor.execute(CREATE_TABLE)
                cursor.execute(
                    "SELECT create_hypertable('glove_motion_frames', 'captured_at', if_not_exists => TRUE)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS glove_motion_frames_session_time_idx "
                    "ON glove_motion_frames (session_id, group_name, timestamp_us DESC)"
                )
                for path in paths:
                    batch = []
                    for frame in read_frames(path):
                        batch.append(
                            (
                                frame["session_id"],
                                frame["group_name"],
                                frame["captured_at"],
                                frame["device_id"],
                                frame["timestamp_us"],
                                frame["label"],
                                Jsonb(frame["streams"]),
                            )
                        )
                        if len(batch) == 500:
                            cursor.executemany(INSERT_FRAME, batch)
                            inserted += cursor.rowcount
                            batch.clear()
                    if batch:
                        cursor.executemany(INSERT_FRAME, batch)
                        inserted += cursor.rowcount
    except psycopg.Error as exc:
        raise RuntimeError(f"Tiger Data database operation failed: {exc}") from exc
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "captures",
        nargs="?",
        type=Path,
        default=Path("data") / "captures",
        help="A JSONL capture file or directory of capture files",
    )
    args = parser.parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("Set DATABASE_URL to your Tiger Data PostgreSQL connection string")

    if args.captures.is_file():
        paths = [args.captures]
    elif args.captures.is_dir():
        paths = sorted(args.captures.glob("*.jsonl"))
    else:
        parser.error(f"Capture path does not exist: {args.captures}")
    if not paths:
        parser.error(f"No .jsonl capture files found in {args.captures}")

    try:
        inserted = import_files(database_url, paths)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1
    print(f"Imported {inserted} motion frames from {len(paths)} capture file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
