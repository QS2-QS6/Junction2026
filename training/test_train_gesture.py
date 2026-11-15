import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train_gesture import build_dataset, split_by_session


class GestureDatasetTests(unittest.TestCase):
    def test_builds_fixed_sensor_feature_vectors(self) -> None:
        frames = iter(
            [
                {
                    "session_id": "one",
                    "label": "open",
                    "streams": {
                        "thumb_finger_sensor": {
                            "TRANSFORM@hub": [0, 0, 0, 0, 0, 0, 1]
                        }
                    },
                },
                {
                    "session_id": "two",
                    "label": "closed",
                    "streams": {
                        "index_finger_sensor": {
                            "TRANSFORM@hub": [1, 0, 0, 0, 0, 0, 1]
                        }
                    },
                },
            ]
        )

        sensors, samples = build_dataset(frames)

        self.assertEqual(sensors, ["index_finger_sensor", "thumb_finger_sensor"])
        self.assertEqual(len(samples[0]["features"]), 14)
        self.assertEqual(samples[0]["features"][:7], [0.0] * 7)

    def test_split_requires_multiple_sessions_for_each_label(self) -> None:
        frames = [
            {"session_id": "open-1", "label": "open", "features": []},
            {"session_id": "open-2", "label": "open", "features": []},
            {"session_id": "closed-1", "label": "closed", "features": []},
            {"session_id": "closed-2", "label": "closed", "features": []},
        ]

        train, test, labels = split_by_session(frames, seed=3)

        self.assertEqual(labels, ["closed", "open"])
        self.assertFalse({row["session_id"] for row in train} & {row["session_id"] for row in test})
        self.assertEqual({row["label"] for row in test}, set(labels))

    def test_split_rejects_sessions_with_multiple_action_labels(self) -> None:
        frames = [
            {"session_id": "one", "label": "open", "features": []},
            {"session_id": "one", "label": "closed", "features": []},
            {"session_id": "two", "label": "open", "features": []},
            {"session_id": "three", "label": "closed", "features": []},
        ]

        with self.assertRaisesRegex(ValueError, "exactly one action label"):
            split_by_session(frames, seed=3)


if __name__ == "__main__":
    unittest.main()
