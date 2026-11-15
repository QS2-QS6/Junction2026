import json
import sys
import tempfile
import threading
import types
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path


rgmp_stream = types.ModuleType("rgmp_stream")
rgmp_stream.DATA_FRAME_HEADER_FMT = "<"
rgmp_stream.DATA_FRAME_HEADER_SIZE = 0
rgmp_stream.MSG_PREFIX_DATA = 1
rgmp_stream.MSG_PREFIX_DEFINITION = 2
rgmp_stream.MSG_PREFIX_DISCONNECT = 3
rgmp_stream.OUTER_HEADER_FMT = "<"
rgmp_stream.OUTER_HEADER_SIZE = 0
rgmp_stream.build_group_lookup = lambda *_args: {}
rgmp_stream.recv_exact = lambda *_args: b""
sys.modules["rgmp_stream"] = rgmp_stream

import server
from recording import SessionRecorder


class GloveStateTests(unittest.TestCase):
    def test_definition_exposes_groups_before_first_data_frame(self) -> None:
        state = server.GloveState()

        state.set_definition(12, ["default", "emf_only"])

        snapshot = state.snapshot_json()
        self.assertIn('"device_id":12', snapshot)
        self.assertIn('"default":{"t":null,"s":{}}', snapshot)
        self.assertIn('"emf_only":{"t":null,"s":{}}', snapshot)

    def test_reset_clears_stale_device_data_and_rates(self) -> None:
        state = server.GloveState()
        state.set_definition(12, ["default"])
        state.update_group(12, "default", 100, {"hub": {}})
        state.devices[12]["rates"] = {"default": 83.0}

        state.reset_stream("device disconnected, waiting")

        self.assertIsNone(state.device_id)
        self.assertEqual(state.groups, {})
        self.assertEqual(state.rates, {})
        self.assertEqual(state.status, "device disconnected, waiting")

    def test_disconnect_closes_an_active_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = SessionRecorder(Path(directory))
            state = server.GloveState(recorder)
            state.set_definition(12, ["default"])
            recorder.start("wave", 12)
            state.update_group(12, "default", 100, {"sensor": {}})

            state.reset_stream("device disconnected, waiting")

            result = recorder.snapshot()
            self.assertFalse(result["active"])
            self.assertEqual(result["frames"], 1)

    def test_only_visualizer_page_and_stream_routes_are_registered(self) -> None:
        self.assertEqual(
            server.PAGES,
            {
                "/": "index.html",
                "/index.html": "index.html",
                "/music": "music.html",
                "/music.html": "music.html",
            },
        )

    def test_recording_endpoint_rejects_requests_without_active_device(self) -> None:
        state = server.GloveState(SessionRecorder(Path(tempfile.gettempdir()) / "unused"))
        http_server = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(state, 60))
        thread = threading.Thread(target=http_server.serve_forever)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{http_server.server_port}/api/recording/start",
                data=b'{"label":"open hand"}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as response:
                urllib.request.urlopen(request)
            self.assertEqual(response.exception.code, 409)
            self.assertIn("Connect a Rokoko device", response.exception.read().decode())
        finally:
            http_server.shutdown()
            http_server.server_close()
            thread.join()

    def test_music_simulator_route_serves_the_music_page(self) -> None:
        state = server.GloveState(SessionRecorder(Path(tempfile.gettempdir()) / "unused"))
        http_server = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(state, 60))
        thread = threading.Thread(target=http_server.serve_forever)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{http_server.server_port}/music") as response:
                body = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertIn("<title>Tactile Music</title>", body)
            self.assertIn('id="audioBtn"', body)
        finally:
            http_server.shutdown()
            http_server.server_close()
            thread.join()

    def test_recording_endpoint_saves_labeled_motion_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = server.GloveState(SessionRecorder(Path(directory)))
            state.set_definition(17, ["default"])
            http_server = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(state, 60))
            thread = threading.Thread(target=http_server.serve_forever)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{http_server.server_port}"
                start = urllib.request.Request(
                    f"{base_url}/api/recording/start",
                    data=json.dumps({"label": "open hand"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(start) as response:
                    self.assertEqual(response.status, 200)

                state.update_group(
                    17,
                    "default",
                    1234,
                    {
                        "thumb_finger_sensor": {
                            "TRANSFORM@hub": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                        }
                    },
                )
                stop = urllib.request.Request(
                    f"{base_url}/api/recording/stop", data=b"", method="POST"
                )
                with urllib.request.urlopen(stop) as response:
                    result = json.loads(response.read())["recording"]
                lines = Path(result["file"]).read_text(encoding="utf-8").splitlines()
                frame = json.loads(lines[1])
                self.assertEqual(frame["label"], "open hand")
                self.assertEqual(frame["timestamp_us"], 1234)
            finally:
                http_server.shutdown()
                http_server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
