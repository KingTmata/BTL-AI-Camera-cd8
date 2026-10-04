import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from PIL import Image

from src.classes import PROJECT_NAMES
from src.dataset import lock_release, sha256
from src.training.manager import RunManager, atomic_json, read_json, read_metrics, checkpoint_label
from src.training.progress import TrainingProgress
from src.training.worker import preflight
from src.training.options import ADVANCED_DEFAULTS, mosaic_active_epochs


class TrainingConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "weights").mkdir()
        (self.root / "weights/yolo26n.pt").write_bytes(b"local checkpoint")
        (self.root / "configs/train_n.yaml").write_text(yaml.safe_dump({
            "model": "weights/yolo26n.pt", "data": "configs/data.yaml", "epochs": 50,
            "imgsz": 640, "batch": 8, "optimizer": "MuSGD", "lr0": .001,
            "patience": 10, "seed": 42}))
        self.manager = RunManager(self.root)

    def test_cpu_defaults_keep_optimizer_and_use_full_dataset(self):
        args = self.manager.build_config()
        self.assertEqual((args["epochs"], args["batch"], args["imgsz"], args["workers"]), (1, 2, 640, 0))
        self.assertEqual((args["device"], args["cache"], args["fraction"]), ("cpu", False, 1.0))
        self.assertEqual((args["optimizer"], args["lr0"], args["patience"], args["seed"]), ("MuSGD", .001, 10, 42))
        self.assertTrue(args["save"])
        self.assertEqual(args["save_period"], 1)
        self.assertFalse(args["resume"])

    def test_invalid_inputs_and_checkpoint_escape_are_rejected(self):
        for overrides in ({"epochs": 0}, {"epochs": True}, {"batch": -1}, {"imgsz": 641},
                          {"workers": -1}, {"cache": "disk"}, {"run_name": "../overwrite"},
                          {"run_name": "a; command"}, {"checkpoint": "../outside.pt"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.manager.build_config(**overrides)

    def test_advanced_defaults_and_explicit_overrides_reach_config(self):
        cfg = self.manager.build_config()
        self.assertEqual({key: cfg[key] for key in ADVANCED_DEFAULTS}, ADVANCED_DEFAULTS)
        cfg = self.manager.build_config(advanced={"cls_pw": .25, "mixup": .1, "close_mosaic": 0, "cls": .8})
        self.assertEqual((cfg["cls_pw"], cfg["mixup"], cfg["close_mosaic"], cfg["cls"]), (.25, .1, 0, .8))
        self.assertEqual(cfg["mosaic"], 1.0)
        self.assertEqual(mosaic_active_epochs(4, 10), 0)
        self.assertEqual(mosaic_active_epochs(100, 10), 90)
        self.assertEqual(mosaic_active_epochs(4, 0), 4)

    def test_invalid_advanced_options_are_rejected(self):
        for options in ({"mosaic": 10}, {"cls_pw": 1.1}, {"fliplr": -.1},
                        {"degrees": 181}, {"close_mosaic": 1.5}, {"close_mosaic": True},
                        {"cls": float("nan")}, {"dfl": float("inf")}, {"box": -1},
                        {"mixup": True}, {"unknown": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.manager.build_config(advanced=options)

    def test_partial_state_metrics_and_log_do_not_crash(self):
        state = self.root / "state.json"
        state.write_text('{"status":')
        self.assertIsNone(read_json(state))
        atomic_json(state, {"status": "running"})
        self.assertEqual(read_json(state)["status"], "running")
        metrics = self.root / "results.csv"
        metrics.write_text("epoch,train/box_loss,metrics/mAP50(B)\n1,0.5,0.1\n2,0.")
        self.assertEqual(read_metrics(metrics), [{"epoch": 1., "train/box_loss": .5, "metrics/mAP50(B)": .1}])
        self.assertEqual(checkpoint_label(Path("epoch0.pt")), "Epoch 1 · epoch0.pt")


class TrainingProcessTests(unittest.TestCase):
    def setUp(self):
        TrainingConfigTests.setUp(self)
        self.children = []
        self.addCleanup(self.cleanup_children)
        self.fixture = self.root / "fixture_worker.py"
        self.fixture.write_text('''import json,os,sys,time
from pathlib import Path
run=Path(sys.argv[sys.argv.index("--run")+1])
def state(value):
    p=run/"fixture.tmp"; p.write_text(json.dumps(value)); os.replace(p,run/"state.json")
state({"status":"running","phase":"train"})
while not (run/"FINISH").exists() and not (run/"STOP.json").exists(): time.sleep(.02)
(run/"weights").mkdir(exist_ok=True)
(run/"weights/epoch0.pt").write_bytes(b"checkpoint preserved")
state({"status":"completed" if (run/"FINISH").exists() else "stopped","phase":"done"})
''')

    def cleanup_children(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)

    def spawn_fixture(self, run, token):
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        with (run / "console.log").open("ab") as log:
            child = subprocess.Popen([sys.executable, str(self.fixture), "src.training.worker",
                                      "--run", str(run), "--token", token], stdout=log, stderr=log, **kwargs)
        self.children.append(child)
        return child

    def test_refresh_duplicate_start_and_checkpoint_preservation(self):
        with patch.object(RunManager, "_spawn", side_effect=self.spawn_fixture):
            first = self.manager.start(advanced={"cls_pw": .25, "close_mosaic": 0})
            request = read_json(first / "request.json")
            for cfg in (request["config"], RunManager(self.root).snapshot(first)["config"]):
                self.assertEqual((cfg["cls_pw"], cfg["close_mosaic"], cfg["mosaic"]), (.25, 0, 1.0))
            refreshed = RunManager(self.root)
            self.assertEqual(refreshed.active()["run_id"], first.name)
            with self.assertRaises(ValueError):
                refreshed.start()
            (first / "FINISH").touch()
            self.children[0].wait(timeout=10)
            self.assertIsNone(refreshed.active())
            self.assertEqual(refreshed.snapshot(first)["status"], "completed")
            second = refreshed.start(checkpoint="weights/yolo26n.pt")
            self.assertNotEqual(first, second)
            self.assertEqual((first / "weights/epoch0.pt").read_bytes(), b"checkpoint preserved")
            self.assertEqual((self.root / "weights/yolo26n.pt").read_bytes(), b"local checkpoint")

    def test_dead_process_with_partial_state_is_failed_not_completed(self):
        with patch.object(RunManager, "_spawn", side_effect=self.spawn_fixture):
            run = self.manager.start()
            self.children[0].terminate()
            self.children[0].wait(timeout=10)
            (run / "state.json").write_text('{"status":')
            self.assertIsNone(self.manager.active())
            self.assertEqual(self.manager.snapshot(run)["status"], "failed")

    def test_identity_mismatch_is_stale_and_inference_is_blocked_while_alive(self):
        with patch.object(RunManager, "_spawn", side_effect=self.spawn_fixture):
            run = self.manager.start()
            with self.assertRaises(ValueError), self.manager.inference_slot():
                pass
            record = read_json(self.manager.runs / ".active.json")
            self.assertTrue(self.manager.process_alive(record))
            record["create_time"] -= 30
            self.assertFalse(self.manager.process_alive(record))
            self.manager.request_stop(run)
            self.children[0].wait(timeout=10)
            self.assertIsNone(self.manager.active())
            self.assertEqual(self.manager.snapshot(run)["status"], "stopped")

    def test_concurrent_start_attempts_create_only_one_run(self):
        barrier = threading.Barrier(2)
        results = []
        def start():
            barrier.wait()
            try:
                results.append(RunManager(self.root).start())
            except ValueError:
                results.append(None)
        with patch.object(RunManager, "_spawn", side_effect=self.spawn_fixture):
            threads = [threading.Thread(target=start) for _ in range(2)]
            for thread in threads: thread.start()
            for thread in threads: thread.join(timeout=10)
        self.assertEqual(len([r for r in results if r]), 1)
        self.assertEqual(len(self.manager.history()), 1)

    def test_launch_error_keeps_failed_run_and_releases_claim(self):
        with patch.object(RunManager, "_spawn", side_effect=OSError("cannot launch")):
            with self.assertRaises(ValueError): self.manager.start()
        self.assertIsNone(self.manager.active())
        self.assertEqual(self.manager.history()[0]["status"], "failed")

    def test_ui_poll_waits_for_inference_instead_of_failing_webcam(self):
        acquired = threading.Event()
        results = []
        def poll():
            with RunManager(self.root).guard():
                acquired.set()
                time.sleep(.1)
        thread = threading.Thread(target=poll)
        thread.start()
        self.addCleanup(thread.join, 3)
        self.assertTrue(acquired.wait(3))
        with self.manager.inference_slot():
            results.append("predicted")
        thread.join(3)
        self.assertEqual(results, ["predicted"])


class TrainingCallbackTests(unittest.TestCase):
    def test_stop_waits_for_saved_epoch_and_final_val_does_not_add_an_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            progress = TrainingProgress(run, "token", {"epochs": 3, "batch": 2})
            weights = run / "weights"
            weights.mkdir()
            trainer = SimpleNamespace(epoch=0, stop=False, wdir=weights, last=weights / "last.pt", metrics={"mAP": .1})
            atomic_json(run / "STOP.json", {"token": "token"})
            progress.fit_end(trainer)
            self.assertFalse(trainer.stop, "Stop must wait for checkpoint, even if requested early")
            trainer.last.write_bytes(b"saved")
            (weights / "epoch0.pt").write_bytes(b"saved")
            progress.model_save(trainer)
            progress.fit_end(trainer)
            self.assertTrue(trainer.stop)
            self.assertEqual(progress.state["epochs_completed"], 1)
            progress.val_start(SimpleNamespace(training=False, dataloader=[1]))
            trainer.epoch = 1  # Ultralytics increments epoch temporarily in final_eval.
            progress.fit_end(trainer)
            self.assertEqual(progress.state["epochs_completed"], 1)
            self.assertEqual(progress.state["phase"], "final_val")
            self.assertGreater(progress.state["rss_peak_bytes"], 0)

    def test_partial_csv_and_non_finite_callback_values_are_not_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            progress = TrainingProgress(folder, "token", {"epochs": 1, "batch": 2})
            trainer = SimpleNamespace(metrics={"valid": .2, "bad": float("nan")}, epoch=0)
            progress.fit_end(trainer)
            self.assertEqual(progress.state["val_metrics"], {"valid": .2})
            state = read_json(Path(folder) / "state.json")
            self.assertEqual(state["val_metrics"], {"valid": .2})


class TrainingPreflightTests(unittest.TestCase):
    def setUp(self):
        TrainingConfigTests.setUp(self)
        rows = []
        for i in range(8):
            image = self.root / f"image{i}.png"
            Image.new("RGB", (20, 20), (i*20, 0, 0)).save(image)
            image.with_suffix(".txt").write_text("\n".join(f"{c} .5 .5 .2 .2" for c in range(8)))
            rows.append({"image_id": f"img{i}", "image_path": image.name, "label_path": image.with_suffix(".txt").name,
                         "source_id": "test", "session_id": str(i), "split_group_id": str(i),
                         "split": "train" if i < 4 else "val" if i < 6 else "test", "annotation_status": "approved",
                         "annotator": "source", "reviewer": "human", "review_status": "approved",
                         "near_duplicate_status": "approved", "license_or_consent": "test fixture", "needs_review": "no"})
        self.dataset = self.root / "data/dataset/approved"
        lock_release(rows, self.dataset, self.root)
        marker = self.dataset / "RELEASE.json"
        marker.write_text(json.dumps({"schema_version": 1, "storage": "in_place", "status": "locked", "classes": list(PROJECT_NAMES)}))
        checksums = self.dataset / "checksums.json"
        entries = json.loads(checksums.read_text())
        entries["RELEASE.json"] = sha256(marker)
        checksums.write_text(json.dumps(entries))
        (self.root / "configs/data.yaml").write_bytes((self.dataset / "data.yaml").read_bytes())
        self.request = {"config": self.manager.build_config()}

    def test_preflight_reuses_release_validator_and_correct_mapping(self):
        fake_model = SimpleNamespace(names=dict(enumerate(PROJECT_NAMES)), task="detect")
        with patch("ultralytics.YOLO", return_value=fake_model):
            metadata, model = preflight(self.request, self.root)
        self.assertIs(model, fake_model)
        self.assertEqual(metadata["images"], {"train": 4, "val": 2, "test": 2})
        self.assertEqual(metadata["class_mapping"], {i:i for i in range(8)})
        self.assertEqual(metadata["versions"]["ultralytics"], "8.4.165")

    def test_bad_checksum_never_loads_or_trains_a_model(self):
        next((self.dataset / "labels").rglob("*.txt")).write_text("")
        with patch("ultralytics.YOLO") as model, self.assertRaises(ValueError):
            preflight(self.request, self.root)
        model.assert_not_called()

    def test_bad_advanced_request_never_loads_or_trains_a_model(self):
        self.request["config"]["mosaic"] = 10
        with patch("ultralytics.YOLO") as model, self.assertRaisesRegex(ValueError, "mosaic"):
            preflight(self.request, self.root)
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
