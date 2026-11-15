import json
import tempfile
import unittest
from pathlib import Path

from recording import SessionRecorder


class SessionRecorderTests(unittest.TestCase):
    def test_writes_session_header_and_labeled_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = SessionRecorder(Path(directory))
            started = recorder.start("open hand", 17)
            recorder.record_group(
                17,
                "default",
                1234,
                {"thumb_finger_sensor": {"TRANSFORM@hub": [0.1, 0.2, 0.3]}},
            )
            stopped = recorder.stop()

            self.assertFalse(stopped["active"])
            self.assertEqual(stopped["frames"], 1)
            self.assertEqual(started["session_id"], stopped["session_id"])
            lines = Path(stopped["file"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["record_type"], "session")
            frame = json.loads(lines[1])
            self.assertEqual(frame["label"], "open hand")
            self.assertEqual(frame["group_name"], "default")
            self.assertEqual(frame["timestamp_us"], 1234)

    def test_invalid_frame_stops_capture_and_reports_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = SessionRecorder(Path(directory))
            recorder.start("open hand", 17)
            recorder.record_group(17, "default", 1234, {"sensor": {"bad": [float("nan")]}})

            result = recorder.snapshot()

            self.assertFalse(result["active"])
            self.assertIn("Out of range float values", result["error"])

    def test_requires_action_label_and_prevents_overlapping_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = SessionRecorder(Path(directory))
            with self.assertRaises(ValueError):
                recorder.start("  ", None)

            recorder.start("close hand", None)
            with self.assertRaises(RuntimeError):
                recorder.start("another action", None)
            recorder.stop()


if __name__ == "__main__":
    unittest.main()
