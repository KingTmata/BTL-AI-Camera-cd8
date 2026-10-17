import json
import tempfile
import unittest
from pathlib import Path

from classroom_ai.training.manager import RunManager


class TrainingActiveCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temporary.name)
        self.manager = RunManager(self.root)
        self.run = self.manager.runs / "stale_run"
        self.run.mkdir(parents=True)
        self.token = "a" * 32
        (self.run / "request.json").write_text(
            json.dumps(
                {
                    "run_id": self.run.name,
                    "token": self.token,
                    "config": {},
                }
            ),
            encoding="utf-8",
        )
        (self.run / "state.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
        self.active_path = self.manager.runs / ".active.json"
        self.active_path.write_text(
            json.dumps(
                {
                    "run_dir": str(self.run),
                    "token": self.token,
                    "pid": 12345,
                    "create_time": 1.0,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_stale_active_record_is_removed_after_failure_is_recorded(self):
        self.manager.process_alive = lambda record: False
        with self.manager.guard():
            self.assertIsNone(self.manager._active())
        state = json.loads((self.run / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "failed")
        self.assertFalse(self.active_path.exists())

    def test_live_training_still_blocks_inference(self):
        self.manager.process_alive = lambda record: True
        with self.assertRaisesRegex(ValueError, "Training đang chạy"):
            with self.manager.inference_slot():
                pass

    def test_inference_slot_reports_lock_and_active_check_time(self):
        self.manager.process_alive = lambda record: False
        with self.manager.inference_slot() as timing:
            self.assertGreaterEqual(timing["manager_lock_wait_ms"], 0)
            self.assertGreaterEqual(timing["manager_active_check_ms"], 0)


if __name__ == "__main__":
    unittest.main()
