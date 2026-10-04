"""Observe public Ultralytics callbacks; never replace its training loop."""

import math
import time
from pathlib import Path

import psutil

from src.training.manager import atomic_json, read_json


def numbers(values):
    result = {}
    for key, value in (values or {}).items():
        try:
            number = float(value)
            if math.isfinite(number):
                result[str(key)] = number
        except (ValueError, TypeError):
            pass
    return result


class TrainingProgress:
    def __init__(self, run, token, config):
        self.run, self.token, self.config = Path(run), token, config
        self.started = self.phase_started = time.monotonic()
        self.last_write = 0
        self.durations = {}
        self.final_validation = False
        self.stop_accepted = False
        self.state = {"run_id": self.run.name, "status": "running", "phase": "preflight",
                      "started_at": time.time(), "epoch": 0, "epochs_completed": 0,
                      "epochs_requested": config["epochs"], "batch_size": config["batch"],
                      "batch": 0, "batches": 0, "rss_peak_bytes": 0,
                      "worker_peak_working_set_bytes": 0, "available_ram_min_bytes": psutil.virtual_memory().available}

    def publish(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_write < 1:
            return
        self.last_write = now
        process = psutil.Process()
        info = process.memory_info()
        rss = info.rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                pass
        available = psutil.virtual_memory().available
        self.state.update(worker_pid=process.pid, worker_rss_bytes=info.rss,
                          process_tree_rss_bytes=rss, available_ram_bytes=available,
                          rss_peak_bytes=max(self.state["rss_peak_bytes"], rss),
                          worker_peak_working_set_bytes=max(self.state["worker_peak_working_set_bytes"],
                                                            getattr(info, "peak_wset", info.rss)),
                          available_ram_min_bytes=min(self.state["available_ram_min_bytes"], available),
                          elapsed_seconds=now-self.started, updated_at=time.time(),
                          timings={**self.durations, self.state["phase"]:
                                   self.durations.get(self.state["phase"], 0) + now-self.phase_started})
        atomic_json(self.run / "state.json", self.state)

    def phase(self, name):
        now = time.monotonic()
        old = self.state["phase"]
        self.durations[old] = self.durations.get(old, 0) + now-self.phase_started
        self.phase_started = now
        self.state["phase"] = name
        self.publish(force=True)

    def train_start(self, trainer):
        self.state.update(train_images=len(trainer.train_loader.dataset), val_images=len(trainer.test_loader.dataset),
                          batch_size=trainer.batch_size)
        self.publish(force=True)

    def epoch_start(self, trainer):
        self.final_validation = False
        self.state.update(epoch=trainer.epoch+1, batch=0, batches=len(trainer.train_loader), phase_batch_size=trainer.batch_size)
        self.phase("train")

    def batch_end(self, trainer):
        self.state["batch"] += 1
        self.state["losses"] = numbers(trainer.label_loss_items(trainer.tloss, prefix="train"))
        self.publish(force=self.state["batch"] == self.state["batches"])

    def val_start(self, validator):
        self.final_validation = not validator.training
        self.state.update(batch=0, batches=len(validator.dataloader),
                          phase_batch_size=getattr(validator.dataloader, "batch_size", None))
        self.phase("final_val" if self.final_validation else "val")

    def val_batch_end(self, validator):
        self.state["batch"] += 1
        self.publish(force=self.state["batch"] == self.state["batches"])

    def val_end(self, validator):
        self.state["final_metrics" if self.final_validation else "val_metrics"] = numbers(validator.metrics.results_dict)
        self.phase("checkpoint" if not self.final_validation else "finalizing")

    def model_save(self, trainer):
        expected = trainer.wdir / f"epoch{trainer.epoch}.pt"
        if not expected.is_file() or not trainer.last.is_file():
            raise RuntimeError("Trainer chưa lưu đủ checkpoint của epoch; không xác nhận hoàn thành.")
        self.state["epochs_completed"] = trainer.epoch+1
        self.state["checkpoint_epoch"] = trainer.epoch+1
        self.publish(force=True)

    def fit_end(self, trainer):
        if self.final_validation:
            self.state["final_metrics"] = numbers(trainer.metrics)
        else:
            self.state["val_metrics"] = numbers(trainer.metrics)
            stop = read_json(self.run / "STOP.json")
            if (stop and stop.get("token") == self.token
                    and self.state["epochs_completed"] == trainer.epoch+1):
                # This callback follows save_model; do not set stop during a batch.
                trainer.stop = True
                self.stop_accepted = True
                self.state["stop_after_epoch_accepted"] = True
        self.publish(force=True)

    def callbacks(self):
        return {"on_train_start": self.train_start, "on_train_epoch_start": self.epoch_start,
                "on_train_batch_end": self.batch_end, "on_val_start": self.val_start,
                "on_val_batch_end": self.val_batch_end, "on_val_end": self.val_end,
                "on_model_save": self.model_save, "on_fit_epoch_end": self.fit_end}
