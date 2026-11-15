import json
import tempfile
import unittest
from pathlib import Path

from import_tigerdata import read_frames


class ImportTigerDataTests(unittest.TestCase):
    def test_reads_labeled_frame_after_session_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "record_type": "session",
                                "session_id": "session-1",
                                "label": "wave",
                            }
                        ),
                        json.dumps(
                            {
                                "record_type": "frame",
                                "session_id": "session-1",
                                "label": "wave",
                                "group_name": "default",
                                "captured_at": "2026-01-01T00:00:00+00:00",
                                "timestamp_us": 42,
                                "streams": {"thumb": {"TRANSFORM@hub": [0, 0, 0, 0, 0, 0, 1]}},
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            frame = next(read_frames(path))

            self.assertEqual(frame["session_id"], "session-1")
            self.assertEqual(frame["label"], "wave")
            self.assertEqual(frame["timestamp_us"], 42)

    def test_rejects_frames_before_session_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text('{"record_type":"frame"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "before session header"):
                next(read_frames(path))


if __name__ == "__main__":
    unittest.main()
