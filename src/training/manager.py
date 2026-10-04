"""Local run configuration, process ownership and durable training snapshots."""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psutil
import yaml

from src.training.options import ADVANCED_DEFAULTS, validate_advanced

ROOT = Path(__file__).resolve().parents[2]
TERMINAL = {"completed", "stopped", "failed"}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.05)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None


def read_metrics(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
        # A CSV row still being appended is not a measurement yet.
        if not text.endswith("\n"):
            text = text[:text.rfind("\n") + 1]
        rows = []
        for row in csv.DictReader(text.splitlines()):
            try:
                if None in row:
                    continue
                values = {key.strip(): float(value) for key, value in row.items()}
                if values and all(math.isfinite(v) for v in values.values()):
                    rows.append(values)
            except (ValueError, TypeError):
                continue
        return rows
    except (OSError, UnicodeError, csv.Error):
        return []


def checkpoint_label(path):
    match = re.fullmatch(r"epoch(\d+)\.pt", Path(path).name)
    return f"Epoch {int(match[1]) + 1} · {Path(path).name}" if match else Path(path).name


class RunManager:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.runs = self.root / "runs/train"

    def build_config(self, *, epochs=1, batch=2, imgsz=640, workers=0, cache=False,
                     run_name="project8_v03_26n_cpu", checkpoint=None, advanced=None):
        for name, value, minimum, maximum in (("epochs", epochs, 1, 1000), ("batch", batch, 1, 16),
                                             ("imgsz", imgsz, 64, 1280), ("workers", workers, 0, 4)):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} phải là số nguyên từ {minimum} đến {maximum}.")
        if imgsz % 32 or type(cache) is not bool:
            raise ValueError("imgsz phải chia hết cho 32; cache chỉ nhận bật/tắt RAM.")
        if not isinstance(run_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
            raise ValueError("Tên run dùng 1–64 chữ/số ASCII, dấu gạch dưới hoặc gạch nối.")
        cfg = yaml.safe_load((self.root / "configs/train_n.yaml").read_text(encoding="utf-8"))
        inherited = {key: cfg.get(key, default) for key, default in ADVANCED_DEFAULTS.items()}
        if advanced is not None:
            validate_advanced(advanced)
            inherited.update(advanced)
        cfg.update(validate_advanced(inherited))
        model = (self.root / (checkpoint or cfg["model"])).resolve()
        if (model.suffix != ".pt" or not model.is_file()
                or not any(model.is_relative_to(folder) for folder in (self.root / "weights", self.runs))):
            raise ValueError("Chọn checkpoint .pt có sẵn trong weights hoặc runs/train.")
        cfg.update(model=str(model), data=str(self.root / "configs/data.yaml"),
                   epochs=epochs, batch=batch, imgsz=imgsz, workers=workers, cache=cache,
                   name=run_name, project=str(self.runs), device="cpu", fraction=1.0,
                   save=True, save_period=1, resume=False, exist_ok=True, val=True, nms=False)
        return cfg

    @contextlib.contextmanager
    def guard(self, timeout=5):
        """OS advisory lock: separate Streamlit processes cannot start together."""
        self.runs.mkdir(parents=True, exist_ok=True)
        with (self.runs / ".start.guard").open("a+b") as stream:
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            deadline = time.monotonic() + timeout
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise ValueError("Một phiên khác đang xử lý. Thử lại sau vài giây.") from exc
                    time.sleep(.02)
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def owned_run(self, run):
        run = Path(run).resolve()
        request = read_json(run / "request.json")
        if (run.parent != self.runs or not request or request.get("run_id") != run.name
                or not re.fullmatch(r"[0-9a-f]{32}", request.get("token", ""))):
            raise ValueError("Run không thuộc công cụ training của dự án.")
        return run, request

    @staticmethod
    def process_alive(record):
        try:
            process = psutil.Process(record["pid"])
            args = process.cmdline()
            return (process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                    and abs(process.create_time() - record["create_time"]) < .05
                    and "src.training.worker" in args
                    and args[-4:] == ["--run", record["run_dir"], "--token", record["token"]])
        except (psutil.Error, KeyError, TypeError):
            return False

    def _active(self):
        path = self.runs / ".active.json"
        record = read_json(path)
        if not record:
            if path.exists():
                raise ValueError("Không đọc được khóa tiến trình; chưa thể xác nhận training cũ đã dừng.")
            return None
        run, request = self.owned_run(record["run_dir"])
        if request["token"] != record.get("token"):
            raise ValueError("Khóa training không khớp run; không mở thêm tiến trình.")
        if self.process_alive(record):
            return {**self.snapshot(run), "process_alive": True}
        if record.get("pid") is None and time.time() - record.get("claimed_at", 0) < 30:
            try:
                launcher = psutil.Process(record["launcher_pid"])
                if abs(launcher.create_time() - record["launcher_create_time"]) < .05:
                    return {**self.snapshot(run), "process_alive": True}
            except (psutil.Error, KeyError):
                pass
        state = read_json(run / "state.json") or {}
        if state.get("status") not in TERMINAL:
            atomic_json(run / "state.json", {**state, "run_id": run.name, "status": "failed",
                        "phase": "failed", "error": "Tiến trình đã kết thúc mà chưa xác nhận hoàn tất.",
                        "finished_at": time.time()})
        return None

    def active(self):
        with self.guard():
            return self._active()

    @contextlib.contextmanager
    def inference_slot(self):
        with self.guard():
            if self._active():
                raise ValueError("Training đang chạy. Recognition tạm dừng để dành CPU/RAM.")
            yield

    def _spawn(self, run, token):
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
               "YOLO_AUTOINSTALL": "false", "YOLO_CONFIG_DIR": str(run / "ultralytics_settings")}
        kwargs = {"cwd": self.root, "stdin": subprocess.DEVNULL, "close_fds": True, "env": env}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        with (run / "console.log").open("ab") as log:
            return subprocess.Popen([sys.executable, "-m", "src.training.worker", "--run", str(run),
                                     "--token", token], stdout=log, stderr=log, **kwargs)

    def start(self, **options):
        config = self.build_config(**options)
        with self.guard(timeout=0):
            if self._active():
                raise ValueError("Đã có một run training đang hoạt động trên máy này.")
            token = uuid.uuid4().hex
            timestamp = datetime.now(ZoneInfo("Asia/Saigon")).strftime("%Y%m%d_%H%M%S")
            run = self.runs / f"{config['name']}_{timestamp}_{token[:8]}"
            run.mkdir(exist_ok=False)
            config["name"] = run.name
            request = {"schema_version": 1, "run_id": run.name, "token": token,
                       "root": str(self.root), "created_at": time.time(), "config": config}
            atomic_json(run / "request.json", request)
            atomic_json(run / "state.json", {"run_id": run.name, "status": "starting",
                        "phase": "preflight", "epoch": 0, "epochs_completed": 0, "created_at": request["created_at"]})
            record = {"run_dir": str(run), "token": token, "pid": None, "claimed_at": time.time(),
                      "launcher_pid": os.getpid(), "launcher_create_time": psutil.Process().create_time()}
            atomic_json(self.runs / ".active.json", record)
            try:
                child = self._spawn(run, token)
                record.update(pid=child.pid, create_time=psutil.Process(child.pid).create_time())
                atomic_json(self.runs / ".active.json", record)
            except Exception as exc:
                atomic_json(run / "state.json", {"run_id": run.name, "status": "failed", "phase": "failed",
                            "error": str(exc), "finished_at": time.time()})
                record["claimed_at"] = 0
                atomic_json(self.runs / ".active.json", record)
                raise ValueError(f"Không tạo được tiến trình training: {exc}") from exc
            return run

    def request_stop(self, run):
        run, request = self.owned_run(run)
        with self.guard():
            active = self._active()
            if not active or active["run_id"] != run.name:
                raise ValueError("Run này không còn hoạt động.")
            atomic_json(run / "STOP.json", {"token": request["token"], "requested_at": time.time()})

    def snapshot(self, run):
        run, request = self.owned_run(run)
        state = read_json(run / "state.json") or {"status": "starting", "phase": "đang đọc trạng thái"}
        try:
            with (run / "console.log").open("rb") as stream:
                stream.seek(max(0, stream.seek(0, 2) - 24000))
                log = stream.read().decode("utf-8", errors="replace")
        except OSError:
            log = ""
        checkpoints = [str(p) for p in sorted((run / "weights").glob("*.pt")) if p.is_file()]
        return {**state, "run_id": run.name, "run_dir": str(run), "config": request["config"],
                "metrics_rows": read_metrics(run / "results.csv"), "log_tail": log,
                "checkpoints": checkpoints, "stop_requested": (run / "STOP.json").exists()}

    def history(self):
        return [self.snapshot(p) for p in sorted(self.runs.glob("*"), reverse=True)
                if p.is_dir() and read_json(p / "request.json")]
