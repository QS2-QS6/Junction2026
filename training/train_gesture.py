"""Train a labeled gesture classifier from local captures or Tiger Data."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterator


def read_local_frames(capture_dir: Path, group_name: str) -> Iterator[dict[str, Any]]:
    files = sorted(capture_dir.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No JSONL captures found in {capture_dir}")
    for path in files:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if (
                    isinstance(record, dict)
                    and record.get("record_type") == "frame"
                    and record.get("group_name") == group_name
                ):
                    yield record


def read_tigerdata_frames(database_url: str, group_name: str) -> Iterator[dict[str, Any]]:
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "Tiger Data training requires psycopg; install requirements-verda.txt."
        ) from exc

    query = """
        SELECT session_id, label, streams
        FROM glove_motion_frames
        WHERE group_name = %s
        ORDER BY captured_at
    """
    try:
        with psycopg.connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (group_name,))
                for session_id, label, streams in cursor:
                    yield {
                        "session_id": session_id,
                        "label": label,
                        "streams": streams,
                    }
    except psycopg.Error as exc:
        raise RuntimeError(f"Could not read training frames from Tiger Data: {exc}") from exc


def build_dataset(
    frames: Iterator[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    parsed: list[dict[str, Any]] = []
    sensors: set[str] = set()
    for frame in frames:
        streams = frame.get("streams")
        if not isinstance(streams, dict):
            raise ValueError("A captured frame has no valid streams object")
        session_id = frame.get("session_id")
        label = frame.get("label")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("A captured frame is missing its session_id")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Session {session_id} has no action label")
        for sensor, values in streams.items():
            if sensor == "hub":
                continue
            transform = values.get("TRANSFORM@hub") if isinstance(values, dict) else None
            if transform is not None:
                if not isinstance(transform, list) or len(transform) != 7:
                    raise ValueError(f"Sensor {sensor} must have a 7-value TRANSFORM@hub")
                if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in transform):
                    raise ValueError(f"Sensor {sensor} transform contains a non-finite value")
                sensors.add(sensor)
        parsed.append({"session_id": session_id, "label": label.strip(), "streams": streams})

    sensor_ids = sorted(sensors)
    if not sensor_ids:
        raise ValueError("No TRANSFORM@hub sensor streams found in the selected group")
    for frame in parsed:
        features: list[float] = []
        for sensor in sensor_ids:
            sensor_streams = frame["streams"].get(sensor)
            transform = (
                sensor_streams.get("TRANSFORM@hub", [0.0] * 7)
                if isinstance(sensor_streams, dict)
                else [0.0] * 7
            )
            features.extend(float(value) for value in transform)
        frame["features"] = features
    return sensor_ids, parsed


def split_by_session(
    frames: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    sessions_by_label: dict[str, set[str]] = {}
    for frame in frames:
        sessions_by_label.setdefault(frame["label"], set()).add(frame["session_id"])
    labels = sorted(sessions_by_label)
    if len(labels) < 2:
        raise ValueError("Capture at least two distinct action labels to train a classifier")
    session_labels: dict[str, set[str]] = {}
    for frame in frames:
        session_labels.setdefault(frame["session_id"], set()).add(frame["label"])
    mixed_sessions = [session for session, values in session_labels.items() if len(values) > 1]
    if mixed_sessions:
        raise ValueError(
            "Each recording session must have exactly one action label; "
            f"mixed labels found in session {mixed_sessions[0]!r}"
        )

    rng = random.Random(seed)
    test_sessions: set[str] = set()
    for label, sessions in sessions_by_label.items():
        session_ids = sorted(sessions)
        if len(session_ids) < 2:
            raise ValueError(
                f"Action label {label!r} needs at least two separate recording sessions "
                "for a session-level train/test split"
            )
        rng.shuffle(session_ids)
        test_count = max(1, round(len(session_ids) * 0.2))
        test_count = min(test_count, len(session_ids) - 1)
        test_sessions.update(session_ids[:test_count])

    train = [frame for frame in frames if frame["session_id"] not in test_sessions]
    test = [frame for frame in frames if frame["session_id"] in test_sessions]
    return train, test, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--captures", type=Path, default=Path("data") / "captures")
    source.add_argument("--tigerdata", action="store_true", help="Read training frames from DATABASE_URL")
    parser.add_argument("--group", default="default", help="RGMP group to train on")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "gesture_model.pt")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least one")

    try:
        if args.tigerdata:
            database_url = os.environ.get("DATABASE_URL")
            if not database_url:
                parser.error("--tigerdata requires DATABASE_URL")
            frames = read_tigerdata_frames(database_url, args.group)
        else:
            frames = read_local_frames(args.captures, args.group)
        sensor_ids, samples = build_dataset(frames)
        train, test, labels = split_by_session(samples, args.seed)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 1

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        print(
            "Training requires PyTorch; install requirements-verda.txt in the Verda environment.",
            file=sys.stderr,
        )
        return 1

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_ids = {label: index for index, label in enumerate(labels)}
    train_x = torch.tensor([row["features"] for row in train], dtype=torch.float32, device=device)
    train_y = torch.tensor([label_ids[row["label"]] for row in train], dtype=torch.long, device=device)
    test_x = torch.tensor([row["features"] for row in test], dtype=torch.float32, device=device)
    test_y = torch.tensor([label_ids[row["label"]] for row in test], dtype=torch.long, device=device)
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0, unbiased=False).clamp(min=1e-6)
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale

    model = nn.Sequential(
        nn.Linear(train_x.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, len(labels)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    batch_size = min(256, len(train))
    for _epoch in range(args.epochs):
        model.train()
        for indices in torch.randperm(len(train_x), device=device).split(batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        accuracy = (model(test_x).argmax(dim=1) == test_y).float().mean().item()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "sensor_ids": sensor_ids,
            "labels": labels,
            "mean": mean.cpu(),
            "scale": scale.cpu(),
            "group": args.group,
        },
        args.output,
    )
    print(
        f"Saved {args.output}; {len(train)} train frames, {len(test)} held-out frames, "
        f"{len(labels)} labels, {device.type.upper()} device, {accuracy:.1%} held-out accuracy."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
