"""One local web entry over the existing video engine; no model in HTTP handlers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import struct
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from classroom_ai.paths import PROJECT_ROOT
from classroom_ai.video.actions import (
    LABELS,
    RaiseHandState,
    aggregate_events,
    propose_raise_hand_events,
)
from classroom_ai.video.core import VideoConfig
from classroom_ai.video.pipeline import run_video
from classroom_ai.video.pose_quality import PoseQualityGate
from classroom_ai.video.pose_rules import BasicPoseActionState
from classroom_ai.video.review import MAX_PILOT_WINDOWS, audit_pilot_review, validate_review

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
TERMINAL = {"complete", "failed", "cancelled"}
PREVIEW_CONTRACT = 1
MAX_FRAME_RANGE_BYTES = 4 * 1024 * 1024
MAX_CLASS_WINDOWS = 500
MAX_LIVE_BYTES = 4 * 1024 * 1024
MAX_PILOT_MANIFESTS = 64
MAX_PILOT_MANIFEST_BYTES = 1024 * 1024
PREVIEW_WARNING = "Bản xem này chưa xác minh căn chỉnh PTS/hướng ảnh. Chỉ tải/xem tệp lịch sử; cần chạy lượt mới để bật overlay và gán nhãn."
REVIEW_NAME = re.compile(r"review-[a-f0-9]{32}\.json")
ARTIFACTS = {
    "manifest.json",
    "frames.jsonl",
    "frames.times",
    "summary.json",
    "events.jsonl",
    "metadata.json",
    "report.json",
    "detections.jsonl",
    "tracks_raw.jsonl",
    "tracks.jsonl",
    "annotated.mp4",
    "actions_preview.mp4",
    "preview.mp4",
    "proposals.json",
    "updates.jsonl",
    "latencies.jsonl",
    "actions.jsonl",
}

# These source snapshots describe the classes loaded by this application.
# Source-code updates require restarting the application; do not fingerprint
# new disk bytes while continuing to execute previously imported rule classes.
ACTION_RULE_IMPLEMENTATION_SOURCES = {
    "classroom_ai/video/" + name: hashlib.sha256(
        (Path(__file__).resolve().parents[1] / "video" / name).read_bytes()
    ).hexdigest()
    for name in ("actions.py", "pose_rules.py", "pose_quality.py")
}
ACTION_RULE_IMPLEMENTATION_SHA256 = hashlib.sha256(
    json.dumps(ACTION_RULE_IMPLEMENTATION_SOURCES, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
ACTION_LABEL_INTEGRITY_SCOPE = (
    "sha256_verified_on_first_read_per_process_or_changed_file_identity; "
    "stable_stat_reuses_digest keyed by file ID, size, mtime_ns and ctime_ns. "
    "Detects ordinary replacement/modification, not continuous cryptographic authentication "
    "against changes preserving every identity/stat field. Rule source snapshots are captured "
    "at application import; code changes require restart. GET never rebuilds artifacts."
)


class HandRuleInput(BaseModel):
    """One cue/quality contract shared by replay, proposals and saved rendering."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    pose_quality: dict | None = None
    hand_cue: Literal["nose_both_arms_v1", "shoulder_any_arm_v1"] = "nose_both_arms_v1"

    @field_validator("pose_quality")
    @classmethod
    def quality_config(cls, value):
        PoseQualityGate(value)
        return value


class JobInput(HandRuleInput):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    source: str
    model_id: str
    analysis_task: str = "detect"
    device: str = "cpu"
    tracking: str = "off"
    track_high_thresh: float | None = Field(default=None, ge=0, le=1, strict=True)
    track_low_thresh: float | None = Field(default=None, ge=0, le=1, strict=True)
    new_track_thresh: float | None = Field(default=None, ge=0, le=1, strict=True)
    recovery_confirm_observations: int = Field(default=2, ge=1, strict=True)
    confidence: float = Field(default=0.05, gt=0, le=1)
    imgsz: int = Field(default=1280, gt=0, le=2560)
    nms_iou: float | None = Field(default=None, ge=0, le=1)
    processing_mode: Literal["full", "replay"] = "full"
    sample_fps: float = Field(default=5, gt=0, le=10)
    action_rules: Literal["off", "raise_hand", "basic_pose"] = "off"
    save_action_video: bool = False
    board_roi: list[float] | None = None
    board_roi_source_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @field_validator("board_roi", mode="before")
    @classmethod
    def board_rectangle(cls, value):
        return ActionLabelsInput.board_rectangle(value)


class ActionLabelsInput(HandRuleInput):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action_rules: Literal["raise_hand", "basic_pose"] = "basic_pose"
    board_roi: list[float] | None = None
    board_roi_source_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @field_validator("board_roi", mode="before")
    @classmethod
    def board_rectangle(cls, value):
        if value is None:
            return None
        if (
            not isinstance(value, list)
            or len(value) != 4
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not 0 <= v <= 1
                for v in value
            )
            or not value[0] < value[2]
            or not value[1] < value[3]
        ):
            raise ValueError(
                "Vùng sàn gần bảng cần [x1,y1,x2,y2] chuẩn hóa 0–1, có diện tích dương."
            )
        return [float(v) for v in value]


class VideoExportInput(ActionLabelsInput):
    start_s: float = Field(default=0, ge=0)
    end_s: float | None = Field(default=None, gt=0)
    stabilize_display: bool = True
    whiteboard_roi: list[float] | None = None
    whiteboard_roi_source_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    board_demo_proximity: bool = False
    whiteboard_offscreen: bool = False

    @field_validator("whiteboard_roi", mode="before")
    @classmethod
    def whiteboard_rectangle(cls, value):
        return ActionLabelsInput.board_rectangle(value)


class AnnotationSourceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Đường dẫn phải nằm trong thư mục dự án.")
    return path


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )
    for attempt in range(5):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            # Windows readers/virus scanners can briefly deny rename sharing.
            # Keep the previous complete snapshot while retrying at most 150 ms.
            if attempt == 4:
                raise
            time.sleep(0.01 * 2**attempt)


def _publish_review(path, value):
    # Encode before touching disk. A partial file never has a public snapshot name.
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
    temporary = path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link publishes the complete inode exclusively on
        # Windows and POSIX; unlike replace/rename it cannot overwrite a snapshot.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _action_file_identity(path, verified=None):
    """Stream a digest with stable file metadata; no pose/frame history retained."""

    def signature(stat):
        return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino]

    key, before = str(path.resolve()), path.stat()
    identity = signature(before)
    cached = verified.get(key) if verified is not None else None
    if cached is not None and cached["stat"] == identity:
        return {"sha256": cached["sha256"], "stat": identity}
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if signature(before) != signature(after):
        raise ValueError("Dữ liệu nguồn nhãn hành động thay đổi trong lúc đọc.")
    result = {"sha256": digest, "stat": signature(after)}
    if verified is not None:
        verified.pop(key, None)
        verified[key] = result
        while len(verified) > 12:
            del verified[next(iter(verified))]
    return {"sha256": digest, "stat": list(result["stat"])}


def _action_labels_state(mode, board_roi, pose_quality=None, hand_cue="nose_both_arms_v1"):
    if mode == "raise_hand":
        return RaiseHandState(pose_quality=pose_quality, hand_cue=hand_cue)
    return BasicPoseActionState(board_roi=board_roi, pose_quality=pose_quality, hand_cue=hand_cue)


class DemoJobs:
    def __init__(self, root, runner, replay_runner=None):
        self.root, self.runner = root, runner
        self.replay_runner = replay_runner
        self.export_runner = None
        self.annotation_runner = None
        self.directory = root / "runs" / "demo"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demo-job")
        self.cached = None
        self.cache_key = None
        self.closed = False
        self.models = {}
        candidates = [
            root / "LIGHT_CONTRAST_rsLoRA_best_E1.pt",
            *sorted((root / "weights").glob("*.pt")),
        ]
        for path in candidates:
            if path.is_file():
                model_id = hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest()[:16]
                self.models[model_id] = {
                    "id": model_id,
                    "name": path.name,
                    "task": "pose" if "pose" in path.stem.lower() else "detect",
                    "path": path,
                }

    def _manifest_source_hash(self, job):
        directory = job.get("directory")
        if directory is None:
            return None
        try:
            path = directory / "manifest.json"
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            cached = job.get("_manifest_source_hash")
            if cached is not None and cached[0] == signature:
                return cached[1]
            manifest = _json(path)
            value = manifest.get("source_sha256") if isinstance(manifest, dict) else None
            if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
                value = None
            job["_manifest_source_hash"] = (signature, value)
            return value
        except (OSError, ValueError, TypeError):
            return None

    def public(self, job):
        with self.lock:
            directory = job.get("directory")
            artifacts = (
                sorted(
                    p.name
                    for p in directory.iterdir()
                    if p.is_file() and (p.name in ARTIFACTS or REVIEW_NAME.fullmatch(p.name))
                )
                if directory
                else []
            )
            source = job.get("source_meta")
            verified = self.preview_verified(job)
            replay = job.get("processing_mode") == "replay"
            eligible, export_reason = self.video_export_eligibility(job)
            return {k: job.get(k) for k in ("id", "state", "message", "error")} | {
                "artifacts": artifacts,
                "summary": job.get("summary"),
                "source": source,
                "source_sha256": self._manifest_source_hash(job),
                "cancel_requested": job["stop"].exists(),
                "preview_contract": job.get("preview_contract"),
                "preview_verified": verified,
                "preview_warning": None if verified or replay else PREVIEW_WARNING,
                "processing_mode": job.get("processing_mode", "full"),
                "kind": job.get("kind"),
                "source_name": job.get("source_name"),
                "inference_status": "not_run"
                if job.get("processing_mode") == "annotation_prep"
                else None,
                "preparation_progress": deepcopy(job.get("preparation_progress")),
                "replay_settings": job.get("replay_settings"),
                "video_export": deepcopy(job.get("video_export")),
                "video_export_eligible": eligible,
                "video_export_reason": export_reason,
            }

    @staticmethod
    def export_active(job):
        return (job.get("video_export") or {}).get("state") in {"queued", "running"}

    def busy(self):
        return any(j["state"] not in TERMINAL or self.export_active(j) for j in self.jobs.values())

    def video_export_context(self, job):
        if job["state"] != "complete" or job.get("processing_mode", "full") != "full":
            raise ValueError(
                "Xuất MP4 có nhãn cần một lượt Full đã hoàn tất; Replay chỉ có mẫu lấy thưa."
            )
        if not self.preview_verified(job):
            raise ValueError(PREVIEW_WARNING)
        directory = job["directory"]
        manifest = _json(directory / "manifest.json")
        config, source = manifest.get("config", {}), manifest.get("source", {})
        if (
            manifest.get("status") != "completed"
            or config.get("analysis_task") != "pose"
            or config.get("tracking") != "bytetrack"
            or not (directory / "frames.jsonl").is_file()
        ):
            raise ValueError("Xuất video hành động cần kết quả Full/Pose có ByteTrack đã lưu.")
        source_hash = manifest.get("source_sha256", "")
        duration = source.get("duration_seconds")
        if not isinstance(source_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", source_hash):
            raise ValueError("Kết quả thiếu SHA256 nguồn hợp lệ.")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError("Thời lượng nguồn chưa hợp lệ.")
        source_path = source.get("path") or getattr(job.get("config"), "source", None)
        if source_path is None:
            raise ValueError("Không tìm thấy đường dẫn video nguồn đã xử lý.")
        source_path = _inside(self.root, source_path)
        if not source_path.is_file():
            raise ValueError("Video nguồn không còn trên máy.")
        return directory, source_path, source_hash, float(duration)

    def video_export_eligibility(self, job):
        try:
            self.video_export_context(job)
            return True, None
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            return False, str(exc)

    def preview_verified(self, job):
        directory = job.get("directory")
        return bool(
            job.get("processing_mode", "full") in {"full", "annotation_prep"}
            and job.get("preview_contract") == PREVIEW_CONTRACT
            and directory
            and (directory / "preview.mp4").is_file()
        )

    def require_preview(self, job):
        if job.get("processing_mode") == "replay":
            raise HTTPException(
                409, "Replay lấy mẫu không dùng để rà soát hoặc tạo GT; chạy full để gán nhãn."
            )
        if not self.preview_verified(job):
            raise HTTPException(409, PREVIEW_WARNING)

    def update(self, job, **values):
        with self.lock:
            job.update(values)
            _write_json(job["home"] / "job.json", self.public(job))

    def remember(self, job):
        with self.lock:
            # Both new jobs and disk hydration share the same bounded registry.
            # Eviction releases cached track extents, never an active worker.
            self.jobs[job["id"]] = job
            while len(self.jobs) > 20:
                oldest = next(
                    (
                        key
                        for key, value in self.jobs.items()
                        if value["state"] in TERMINAL and not self.export_active(value)
                    ),
                    None,
                )
                if oldest is None:
                    raise RuntimeError("Không còn chỗ cho lượt xử lý đang hoạt động.")
                del self.jobs[oldest]

    def submit(self, request):
        if request.save_action_video and not (
            request.processing_mode == "full"
            and request.analysis_task == "pose"
            and request.tracking == "bytetrack"
        ):
            raise HTTPException(422, "Tự lưu video hành động cần Full/Pose và ByteTrack.")
        roi_source = None
        if request.board_roi is not None:
            try:
                source_path = _inside(self.root, request.source)
                roi_source = _action_file_identity(source_path)
                if request.board_roi_source_sha256 != roi_source["sha256"]:
                    raise ValueError("Vùng bảng không được xác nhận cho SHA256 video đang chọn.")
            except (OSError, ValueError) as exc:
                raise HTTPException(422, str(exc)) from exc
        with self.lock:
            if self.closed or self.busy():
                raise HTTPException(409, "Đang có một lượt xử lý; hãy chờ hoặc hủy lượt đó.")
            try:
                source = _inside(self.root, request.source)
                if not source.is_file() or source.suffix.lower() not in VIDEO_SUFFIXES:
                    raise ValueError("Không thấy video local hợp lệ.")
                model = self.models.get(request.model_id)
                if not model or model["task"] != request.analysis_task:
                    raise ValueError("Model không khớp chế độ Detect/Pose.")
                config = VideoConfig(
                    source,
                    "full",
                    weights=model["path"],
                    device=request.device,
                    analysis_task=request.analysis_task,
                    tracking=request.tracking,
                    track_high_thresh=request.track_high_thresh,
                    track_low_thresh=request.track_low_thresh,
                    new_track_thresh=request.new_track_thresh,
                    recovery_confirm_observations=request.recovery_confirm_observations,
                    confidence=request.confidence,
                    imgsz=request.imgsz,
                    nms_iou=request.nms_iou,
                    person_only=True,
                    no_display=True,
                    allow_vfr=True,
                )
                if request.device not in ("cpu", "auto", "intel:gpu") and not (
                    request.device.startswith("cuda:") and request.device[5:].isdigit()
                ):
                    raise ValueError("Thiết bị phải là cpu, auto, intel:gpu (Pose) hoặc cuda:N.")
                config.validate()
                if request.processing_mode != "replay" and request.action_rules != "off":
                    raise ValueError(
                        "Luật giơ tay tuần tự chỉ dùng trong replay Pose có ByteTrack."
                    )
                if request.processing_mode == "replay":
                    from classroom_ai.video.online import ReplayConfig

                    config = ReplayConfig(
                        source=source,
                        weights=model["path"],
                        device=request.device,
                        analysis_task=request.analysis_task,
                        imgsz=request.imgsz,
                        confidence=request.confidence,
                        sample_fps=request.sample_fps,
                        tracking=request.tracking,
                        track_high_thresh=request.track_high_thresh,
                        track_low_thresh=request.track_low_thresh,
                        new_track_thresh=request.new_track_thresh,
                        recovery_confirm_observations=request.recovery_confirm_observations,
                        nms_iou=request.nms_iou,
                        action_rules=request.action_rules,
                        pose_quality=request.pose_quality,
                        hand_cue=request.hand_cue,
                        board_roi=request.board_roi,
                        board_roi_source_sha256=request.board_roi_source_sha256,
                    )
                    config.validate()
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            job_id = uuid.uuid4().hex
            # Only the newest job retains a live image; terminal histories stay on disk.
            for previous in self.jobs.values():
                previous.pop("_live_pair", None)
                previous.pop("_live_pending", None)
            home = self.directory / "jobs" / job_id
            home.mkdir(parents=True)
            config.output_dir = home / "output"
            job = {
                "id": job_id,
                "home": home,
                "stop": home / "stop.flag",
                "config": config,
                "processing_mode": request.processing_mode,
                "replay_settings": {
                    "sample_fps": request.sample_fps,
                    "tracking": request.tracking,
                    "action_rules": request.action_rules,
                    "pose_quality": request.pose_quality,
                    "hand_cue": request.hand_cue,
                    "board_roi": request.board_roi,
                    "board_roi_source_sha256": request.board_roi_source_sha256,
                }
                if request.processing_mode == "replay"
                else None,
                "state": "queued",
                "message": "Đã xếp lượt xử lý",
                "error": None,
                "_auto_video_export": VideoExportInput(
                    pose_quality=request.pose_quality,
                    hand_cue=request.hand_cue,
                    board_roi=request.board_roi,
                    board_roi_source_sha256=request.board_roi_source_sha256,
                ).model_dump()
                if request.save_action_video
                else None,
            }
            self.remember(job)
            self.update(job)
            self.executor.submit(self.work, job)
            return self.public(job)

    def submit_annotation_source(self, request):
        try:
            source = _inside(self.root, request.source)
            if not source.is_file() or source.suffix.lower() not in VIDEO_SUFFIXES:
                raise ValueError("Không thấy video gốc hợp lệ trong dự án.")
        except (OSError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        with self.lock:
            if self.closed or self.busy():
                raise HTTPException(
                    409, "Đang xử lý hoặc xuất video; chờ hoàn tất trước khi chuẩn bị nguồn mới."
                )
            job_id = uuid.uuid4().hex
            home = self.directory / "jobs" / job_id
            home.mkdir(parents=True)
            job = {
                "id": job_id,
                "home": home,
                "stop": home / "stop.flag",
                "annotation_source": source,
                "source_name": source.name,
                "kind": "annotation_source_preparation",
                "processing_mode": "annotation_prep",
                "state": "queued",
                "message": "Đã xếp chuẩn bị nguồn gán nhãn; không chạy AI",
                "error": None,
            }
            self.remember(job)
            self.update(job)
            self.executor.submit(self.work_annotation_source, job)
            return self.public(job)

    def work_annotation_source(self, job):
        from classroom_ai.video.media import MediaCancelled, prepare_annotation_source

        def cancelled():
            return self.closed or job["stop"].exists()

        last_progress = [0.0, None]

        def progress(value):
            now, phase = time.monotonic(), value.get("phase")
            if now - last_progress[0] < 0.25 and phase == last_progress[1]:
                return
            last_progress[:] = [now, phase]
            count, total = value.get("frames_written"), value.get("frames_total")
            self.update(
                job,
                state="cancelling" if cancelled() else "processing",
                message=f"Chuẩn bị nguồn gán nhãn · {phase} · {count or 0}/{total or '?'} frame · không chạy AI",
                preparation_progress=deepcopy(value),
            )

        try:
            if cancelled():
                raise MediaCancelled("Đã hủy trước khi chuẩn bị nguồn.")
            self.update(
                job,
                state="processing",
                message="Đang xác minh video và chuẩn bị timeline, không chạy AI",
            )
            destination = job["home"] / "output" / ("annotation_" + uuid.uuid4().hex)
            runner = self.annotation_runner or prepare_annotation_source
            directory = Path(
                runner(
                    job["annotation_source"], destination, cancel=cancelled, on_progress=progress
                )
            )
            manifest, metadata, report = (
                _json(directory / name)
                for name in ("manifest.json", "metadata.json", "report.json")
            )
            if (
                directory.resolve() != destination.resolve()
                or manifest.get("kind") != "annotation_source_preparation"
                or manifest.get("status") != "completed"
                or report.get("status") != "completed"
                or manifest.get("inference", {}).get("status") != "not_run"
            ):
                raise ValueError("Bộ chuẩn bị chưa xác nhận nguồn/timeline hoàn chỉnh.")
            with self.lock:
                if cancelled():
                    raise MediaCancelled("Đã hủy trước khi công bố nguồn gán nhãn.")
                self.update(
                    job,
                    directory=directory,
                    source_meta=metadata["source"],
                    summary=report,
                    preview_contract=PREVIEW_CONTRACT,
                    state="complete",
                    message="Nguồn đã sẵn sàng gán nhãn theo thời gian; không chạy AI",
                )
        except Exception as exc:
            stopped = isinstance(exc, MediaCancelled) or cancelled()
            self.update(
                job,
                state="cancelled" if stopped else "failed",
                error=None if stopped else str(exc),
                message="Đã hủy chuẩn bị nguồn" if stopped else "Chuẩn bị nguồn gán nhãn thất bại",
            )

    def annotation_sources(self):
        entries = []
        paths = sorted(
            (self.directory / "jobs").glob("*/job.json"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        for path in paths:
            if len(entries) >= 200:
                break
            if not re.fullmatch(r"[a-f0-9]{32}", path.parent.name):
                continue
            try:
                saved = _json(path)
                if saved.get("processing_mode") != "annotation_prep":
                    continue
                # Use the same restart recovery as the individual job endpoint;
                # a persisted queued/running state has no worker after restart.
                saved = self.public(self.find(path.parent.name))
                source = saved.get("source") or {}
                entries.append(
                    {
                        "job_id": saved["id"],
                        "name": saved.get("source_name") or Path(source.get("path", "Video")).name,
                        "source_sha256": saved.get("source_sha256"),
                        "state": saved.get("state"),
                        "duration_seconds": source.get("duration_seconds"),
                        "review_url": "/?job=" + saved["id"],
                    }
                )
            except (OSError, ValueError, TypeError, KeyError):
                continue
        return {"sources": entries}

    def predictor(self, task, weights, device):
        # A checkpoint may be replaced at the same pathname between jobs.
        # Hash once per model request, outside the per-frame inference loop.
        with weights.open("rb") as stream:
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        key = (task, weights, device, source_hash)
        if self.cache_key != key:
            self.cached = None
            self.cache_key = None
            if task == "pose":
                from classroom_ai.inference.pose import PoseInspector

                inspector = PoseInspector
            else:
                from classroom_ai.inference.detector import ImageInspector

                inspector = ImageInspector
            self.cached = inspector(weights, device)
            self.cache_key = key
        return self.cached

    def work(self, job):
        if job.get("processing_mode") == "replay":
            self.work_replay(job)
            return
        config = job["config"]
        try:
            if job["stop"].exists():
                self.update(job, state="cancelled", message="Đã hủy trước khi chạy")
                return
            self.update(job, state="loading", message="Đang kiểm video và nạp model")
            if self.runner is run_video:
                import torch

                torch.set_num_threads(min(4, os.cpu_count() or 1))

            def progress(message):
                state = job["state"]
                if "Processing full" in message or "Detecting and tracking" in message:
                    state = "processing"
                elif "Rendering" in message or "Finalizing" in message:
                    state = "rendering"
                if job["stop"].exists():
                    state = "cancelling"
                self.update(job, state=state, message=message)

            directory = self.runner(
                config,
                progress=progress,
                stop_file=job["stop"],
                predictor_factory=lambda w, d: self.predictor(config.analysis_task, w, d),
            )
            if directory:
                directory = Path(directory)
                metadata, report = (
                    _json(directory / "metadata.json"),
                    _json(directory / "report.json"),
                )
                self.update(
                    job, directory=directory, source_meta=metadata.get("source"), summary=report
                )
            if job["stop"].exists():
                self.update(job, state="cancelled", message="Đã hủy; kết quả còn lại là partial")
                return
            if not directory or report["status"] != "completed":
                raise RuntimeError("Pipeline chưa hoàn tất; không công bố video hoàn tất.")
            self.update(
                job, state="rendering", message="Đang chuẩn bị video xem và các lớp overlay"
            )
            self.preview(job)
            automatic_export = None
            with self.lock:
                if job["stop"].exists():
                    self.update(job, state="cancelled", message="Đã hủy khi dựng bản xem")
                else:
                    self.update(
                        job,
                        state="complete",
                        message="Đã xử lý xong; tua/bật lớp không chạy lại AI",
                    )
                    if job.get("_auto_video_export"):
                        try:
                            automatic_export = self.reserve_video_export(
                                job, VideoExportInput(**job["_auto_video_export"])
                            )
                        except Exception as exc:
                            self.update(
                                job,
                                video_export={
                                    "id": uuid.uuid4().hex,
                                    "state": "failed",
                                    "error": str(exc),
                                    "video_url": None,
                                    "report_url": None,
                                    "options": job["_auto_video_export"],
                                    "progress": None,
                                },
                            )
            if automatic_export:
                try:
                    self.dispatch_video_export(job, automatic_export)
                except Exception as exc:
                    # Export failure never invalidates completed inference or human review.
                    self.update_video_export(
                        job, automatic_export[0], state="failed", error=str(exc)
                    )
        except Exception as exc:
            outputs = list(config.output_dir.glob("*")) if config.output_dir.exists() else []
            directory = next((p for p in outputs if p.is_dir()), None)
            with self.lock:
                cancelled = job["stop"].exists()
                self.update(
                    job,
                    state="cancelled" if cancelled else "failed",
                    error=None if cancelled else str(exc),
                    message="Đã hủy khi xử lý media" if cancelled else "Lượt xử lý thất bại",
                    directory=directory,
                )

    def live_callbacks(self, job):
        def encoded(row):
            sequence = row.get("sequence") if isinstance(row, dict) else None
            if type(sequence) is not int or sequence < 0:
                raise ValueError("Replay update requires a nonnegative integer sequence.")
            body = json.dumps(
                row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            if len(body) > MAX_LIVE_BYTES:
                raise ValueError("Replay metadata exceeds the 4 MiB limit.")
            return body

        def on_frame(jpeg, row):
            if not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= MAX_LIVE_BYTES:
                raise ValueError("Replay JPEG exceeds the 4 MiB limit or is empty.")
            body = encoded(row)
            with self.lock:
                if job["stop"].exists() or job["state"] in TERMINAL:
                    return
                previous = job.get("_live_pair")
                if previous and row["sequence"] <= previous["sequence"]:
                    raise ValueError(
                        "Replay sequence must increase; duplicate/old update rejected."
                    )
                job["_live_pending"] = (jpeg, body)

        def on_update(row):
            body = encoded(row)
            with self.lock:
                if job["stop"].exists() or job["state"] in TERMINAL:
                    job.pop("_live_pending", None)
                    return
                pending = job.pop("_live_pending", None)
                if pending is None or pending[1] != body:
                    raise ValueError(
                        "Replay JPEG and update do not share the same sequence/source."
                    )
                job["_live_pair"] = {
                    "sequence": row["sequence"],
                    "update": json.loads(body),
                    "jpeg": pending[0],
                }
                job["state"] = "processing"
                job["message"] = "Replay nguồn ghi sẵn 1×; đang công bố mẫu Pose, không phải camera"

        return on_frame, on_update

    def work_replay(self, job):
        from classroom_ai.training.manager import RunManager
        from classroom_ai.video.online import run_replay

        stop, done = threading.Event(), threading.Event()
        job["_replay_stop"] = stop

        def watch():
            while not done.wait(0.05):
                if job["stop"].exists() or self.closed:
                    stop.set()
                    return

        watcher = threading.Thread(target=watch, name="web-replay-cancel")
        watcher.start()
        directory = None
        try:
            if job["stop"].exists():
                stop.set()
            self.update(
                job, state="loading", message="Chuẩn bị replay: kiểm PTS, nạp/export/warmup model"
            )
            on_frame, on_update = self.live_callbacks(job)
            directory = (self.replay_runner or run_replay)(
                job["config"],
                on_frame=on_frame,
                on_update=on_update,
                stop=stop,
                predictor_factory=lambda w, d: self.predictor("pose", w, d),
                manager=RunManager(self.root),
            )
            directory = Path(directory)
            manifest, report = _json(directory / "manifest.json"), _json(directory / "report.json")
            status = report.get("status")
            if status not in ("completed", "cancelled", "failed"):
                raise ValueError("Replay report has no valid terminal status.")
            cancelled = job["stop"].exists() or status == "cancelled"
            self.update(
                job,
                directory=directory,
                source_meta=manifest.get("source"),
                summary=report,
                state="cancelled"
                if cancelled
                else "complete"
                if status == "completed"
                else "failed",
                error=report.get("error") if status == "failed" else None,
                message="Đã hủy replay"
                if cancelled
                else "Replay kết thúc; mẫu/độ trễ máy chủ đã lưu",
            )
        except Exception as exc:
            outputs = list(job["config"].output_dir.glob("*"))
            directory = next((p for p in outputs if p.is_dir()), None)
            cancelled = job["stop"].exists()
            self.update(
                job,
                directory=directory,
                state="cancelled" if cancelled else "failed",
                error=None if cancelled else str(exc),
                message="Đã hủy replay" if cancelled else "Replay lỗi",
            )
        finally:
            done.set()
            watcher.join(timeout=1)
            with self.lock:
                job.pop("_live_pending", None)
                job.pop("_replay_stop", None)

    def latest(self, job):
        if job.get("processing_mode") != "replay":
            raise HTTPException(409, "Latest update chỉ dành cho recorded replay.")
        with self.lock:
            pair = job.get("_live_pair")
            return {
                "state": job["state"],
                "processing_mode": "replay",
                "sequence": pair["sequence"] if pair else None,
                "update": pair["update"] if pair else None,
            }

    def preview(self, job):
        from classroom_ai.video.media import MediaCancelled, VideoWriter, frames, probe_source
        from classroom_ai.video.metrics import Metrics

        # Force exact source PTS even for coarse-clock CFR. Re-encoding decoded
        # raster pixels strips source display rotation/SAR without rotating boxes.
        cancel = job["stop"].exists

        def progress(*_):
            if cancel():
                raise MediaCancelled("Đã hủy quét video xem trước.")

        source = replace(
            probe_source(job["config"].source, allow_vfr=True, cancel=cancel, on_progress=progress),
            vfr=True,
        )
        private = job["directory"] / (".preview-" + uuid.uuid4().hex)
        private.mkdir()
        writer = None
        decoded = None
        try:
            writer = VideoWriter(source, private)
            decoded = frames(source, Metrics(), threading.Event(), cancel=cancel)
            for frame in decoded:
                if job["stop"].exists():
                    return
                writer.write(frame.image, frame=frame)
            if job["stop"].exists():
                return
            completed = writer.finish(cancel=cancel)
            if not job["stop"].exists():
                os.link(completed, job["directory"] / "preview.mp4")
                self.update(job, preview_contract=PREVIEW_CONTRACT)
        finally:
            if decoded is not None:
                decoded.close()
            if writer is not None:
                writer.close()
            shutil.rmtree(private)

    def find(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None and re.fullmatch(r"[a-f0-9]{32}", job_id):
            home = self.directory / "jobs" / job_id
            state_file = home / "job.json"
            if state_file.is_file():
                saved = _json(state_file)
                directory = next((p for p in (home / "output").glob("*") if p.is_dir()), None)
                job = {
                    **saved,
                    "home": home,
                    "stop": home / "stop.flag",
                    "directory": directory,
                    "source_meta": saved.get("source"),
                }
                if job["state"] not in TERMINAL:
                    job.update(state="failed", error="Dịch vụ đã dừng giữa lượt; kết quả partial.")
                interrupted_export = self.export_active(job)
                if interrupted_export:
                    job["video_export"] = {
                        **job["video_export"],
                        "state": "failed",
                        "error": "Dịch vụ đã dừng giữa lúc xuất video; hãy xuất lại.",
                        "video_url": None,
                        "report_url": None,
                    }
                with self.lock:
                    job = self.jobs.get(job_id) or job
                    self.remember(job)
                    if interrupted_export:
                        self.update(job)
        if job is None:
            raise HTTPException(404, "Không thấy lượt xử lý.")
        return job

    def update_video_export(self, job, export_id, **values):
        with self.lock:
            operation = job.get("video_export")
            if not operation or operation["id"] != export_id:
                return
            self.update(job, video_export={**operation, **values})

    def start_video_export(self, job, request):
        return self.dispatch_video_export(job, self.reserve_video_export(job, request))

    def reserve_video_export(self, job, request):
        try:
            directory, source, source_hash, duration = self.video_export_context(job)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(409, str(exc)) from exc
        if request.start_s >= duration or (
            request.end_s is not None and not request.start_s < request.end_s <= duration
        ):
            raise HTTPException(422, "Khoảng xuất cần 0 ≤ bắt đầu < kết thúc ≤ thời lượng nguồn.")
        if request.board_roi is not None and request.board_roi_source_sha256 != source_hash:
            raise HTTPException(422, "Vùng bảng cần SHA256 khớp video đang xuất.")
        options = request.model_dump()
        with self.lock:
            if self.closed or self.busy():
                raise HTTPException(
                    409, "Đang xử lý hoặc xuất video; hãy chờ hoặc dừng lượt hiện tại."
                )
            export_id = uuid.uuid4().hex
            operation = {
                "id": export_id,
                "state": "queued",
                "error": None,
                "options": options,
                "progress": None,
                "cancel_requested": False,
                "video_url": None,
                "report_url": None,
            }
            stop = threading.Event()
            job["_video_export_stop"] = stop
            self.update(job, video_export=operation)
        return export_id, directory, source, source_hash, options, stop

    def dispatch_video_export(self, job, reservation):
        export_id, directory, source, source_hash, options, stop = reservation
        # Reserve the operation first, then fingerprint outside the registry lock.
        try:
            fingerprints = {
                name: _action_file_identity(directory / name)["sha256"]
                for name in ("manifest.json", "frames.jsonl")
            }
            with self.lock:
                future = self.executor.submit(
                    self.work_video_export,
                    job,
                    export_id,
                    directory,
                    source,
                    source_hash,
                    options,
                    fingerprints,
                    stop,
                )
                job["_video_export_future"] = future
        except Exception as exc:
            self.update_video_export(job, export_id, state="failed", error=str(exc))
            raise HTTPException(422, "Không chuẩn bị được đầu vào xuất video: " + str(exc)) from exc
        with self.lock:
            return deepcopy(job["video_export"])

    def work_video_export(
        self, job, export_id, directory, source, source_hash, options, fingerprints, stop
    ):
        from classroom_ai.video.media import MediaCancelled

        last_progress = [0.0, None]

        def cancelled():
            return stop.is_set() or self.closed

        def progress(value):
            now = time.monotonic()
            phase = value.get("phase")
            if phase != last_progress[1] or now - last_progress[0] >= 0.25:
                last_progress[:] = [now, phase]
                self.update_video_export(job, export_id, progress=deepcopy(value))

        try:
            if cancelled():
                raise MediaCancelled("Đã hủy trước khi xuất video.")
            self.update_video_export(job, export_id, state="running")
            for name, expected in fingerprints.items():
                if _action_file_identity(directory / name)["sha256"] != expected:
                    raise ValueError("Kết quả nguồn đã thay đổi sau khi yêu cầu xuất video.")
            if _json(directory / "manifest.json").get("source_sha256") != source_hash:
                raise ValueError("SHA256 nguồn thay đổi sau khi yêu cầu xuất video.")
            runner = self.export_runner
            if runner is None:
                from classroom_ai.video.rendering import export_saved_actions

                runner = export_saved_actions
            destination = directory / "video_exports" / export_id
            destination.parent.mkdir(exist_ok=True)
            result = Path(
                runner(
                    directory,
                    destination,
                    source=source,
                    source_title=source.stem,
                    cancel=cancelled,
                    on_progress=progress,
                    **options,
                )
            )
            if (
                result.resolve() != destination.resolve()
                or not (result / "video.mp4").is_file()
                or not (result / "report.json").is_file()
            ):
                raise ValueError("Bộ xuất chưa công bố đầy đủ video và báo cáo.")
            for name, expected in fingerprints.items():
                if _action_file_identity(directory / name)["sha256"] != expected:
                    raise ValueError("Kết quả nguồn thay đổi trong lúc xuất video.")
            base = f"/api/jobs/{job['id']}/video-export/{export_id}"
            with self.lock:
                if cancelled():
                    raise MediaCancelled("Đã hủy trước khi công bố video.")
                # A renderer directory alone is not a backend publication. Each
                # successful generation retains its own commit for historical links.
                _write_json(
                    result / "web-export.json",
                    {"status": "complete", "job_id": job["id"], "export_id": export_id},
                )
                self.update_video_export(
                    job,
                    export_id,
                    state="complete",
                    error=None,
                    video_url=base + "/video.mp4?inline=true",
                    report_url=base + "/report.json",
                    progress={"phase": "complete"},
                )
        except Exception as exc:
            was_cancelled = isinstance(exc, MediaCancelled) or cancelled()
            self.update_video_export(
                job,
                export_id,
                state="cancelled" if was_cancelled else "failed",
                error=None if was_cancelled else str(exc),
                video_url=None,
                report_url=None,
            )

    def cancel_video_export(self, job):
        with self.lock:
            operation = job.get("video_export")
            if self.export_active(job):
                stop = job.get("_video_export_stop")
                if stop is not None:
                    stop.set()
                future = job.get("_video_export_future")
                queued_cancelled = future is not None and future.cancel()
                self.update_video_export(
                    job,
                    operation["id"],
                    cancel_requested=True,
                    **({"state": "cancelled"} if queued_cancelled else {}),
                )
            return deepcopy(job.get("video_export"))

    def _action_labels_inputs(self, job):
        if job["state"] != "complete":
            raise ValueError("Nhãn hành động cần lượt full đã hoàn tất.")
        self.require_preview(job)
        directory = job["directory"]
        inputs = {
            name: _action_file_identity(
                directory / name, job.setdefault("_action_label_fingerprints", {})
            )
            for name in ("manifest.json", "frames.jsonl", "frames.idx")
        }
        manifest = _json(directory / "manifest.json")
        if not isinstance(manifest, dict):
            raise ValueError("Manifest nguồn cần là object.")
        config, source = manifest.get("config", {}), manifest.get("source", {})
        if not isinstance(config, dict) or not isinstance(source, dict):
            raise ValueError("Config/metadata nguồn không hợp lệ.")
        if (
            manifest.get("status") != "completed"
            or config.get("analysis_task") != "pose"
            or config.get("tracking") != "bytetrack"
            or config.get("mode", "full") != "full"
            or re.fullmatch(r"[a-f0-9]{64}", str(manifest.get("source_sha256", ""))) is None
        ):
            raise ValueError("Chỉ tạo gợi ý từ raw Pose + ByteTrack full đã xác minh.")
        count = source.get("frame_count")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or job["source_meta"].get("frame_count") != count
            or inputs["frames.idx"]["stat"][0] != count * 8
        ):
            raise ValueError("Số frame/index nguồn không khớp.")
        base = Fraction(source["time_base"])
        duration = source["duration_seconds"]
        if base <= 0 or isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Clock nguồn không hợp lệ.")
        for key in ("width", "height"):
            if (
                isinstance(source.get(key), bool)
                or not isinstance(source.get(key), int)
                or source[key] <= 0
            ):
                raise ValueError("Kích thước raster nguồn không hợp lệ.")
        namespace = (
            "run:" + str(manifest["run_id"]) if manifest.get("run_id") else "job:" + job["id"]
        )
        return inputs, manifest, namespace

    def _action_labels_cached(self, job, inputs, manifest, namespace, request=None):
        directory = job["directory"]
        meta = _json(directory / "action-labels.meta.json")
        if not isinstance(meta, dict):
            raise ValueError("Metadata cache cần là object.")
        saved_request = ActionLabelsInput(
            action_rules=meta.get("action_rules"),
            board_roi=meta.get("board_roi"),
            board_roi_source_sha256=meta.get("board_roi_source_sha256"),
            pose_quality=meta.get("pose_quality"),
            hand_cue=meta.get("hand_cue", "nose_both_arms_v1"),
        )
        generation = meta.get("generation")
        if (
            meta.get("schema_version") != 1
            or meta.get("kind") != "full_frame_action_labels"
            or not isinstance(generation, str)
            or re.fullmatch(r"[a-f0-9]{32}", generation) is None
            or meta.get("job_id") != job["id"]
            or meta.get("inputs") != inputs
            or meta.get("source_sha256") != manifest["source_sha256"]
            or meta.get("actor_namespace") != namespace
            or meta.get("frame_count") != manifest["source"]["frame_count"]
            or meta.get("raw_frames_sha256") != inputs["frames.jsonl"]["sha256"]
            or meta.get("ground_truth") is not False
            or meta.get("rule_implementation_sha256") != ACTION_RULE_IMPLEMENTATION_SHA256
            or meta.get("rule_implementation_sources") != ACTION_RULE_IMPLEMENTATION_SOURCES
            or meta.get("integrity_scope") != ACTION_LABEL_INTEGRITY_SCOPE
            or saved_request.board_roi is not None
            and saved_request.board_roi_source_sha256 != manifest["source_sha256"]
            or request is not None
            and saved_request != request
            or meta.get("rule_config")
            != _action_labels_state(
                saved_request.action_rules,
                saved_request.board_roi,
                saved_request.pose_quality,
                saved_request.hand_cue,
            ).config
        ):
            raise ValueError("Cache nhãn hành động thiếu, cũ hoặc không khớp nguồn/quy tắc.")
        cache = (directory / ".action-labels-cache" / generation).resolve()
        if not cache.is_relative_to(directory.resolve()):
            raise ValueError("Đường dẫn cache nhãn hành động không hợp lệ.")
        for name in ("frames.jsonl", "frames.idx"):
            if (cache / name).resolve().parent != cache:
                raise ValueError("Đường dẫn file cache không hợp lệ.")
            if (
                _action_file_identity(
                    cache / name, job.setdefault("_action_label_fingerprints", {})
                )
                != meta["derived"][name]
            ):
                raise ValueError("Cache nhãn hành động bị thay đổi hoặc hỏng.")
        if (cache / "frames.idx").stat().st_size != meta["frame_count"] * 8:
            raise ValueError("Index nhãn hành động không hợp lệ.")
        return meta, cache

    def _action_labels_lock(self, job):
        with self.lock:
            return job.setdefault("_action_labels_lock", threading.Lock())

    def prepare_action_labels(self, job, request):
        with self._action_labels_lock(job):
            temporary = None
            try:
                inputs, manifest, namespace = self._action_labels_inputs(job)
                if (
                    request.board_roi is not None
                    and request.board_roi_source_sha256 != manifest["source_sha256"]
                ):
                    raise HTTPException(
                        422, "Vùng bảng không được xác nhận cho SHA256 video đang xem."
                    )
                try:
                    meta, _ = self._action_labels_cached(job, inputs, manifest, namespace, request)
                    return {
                        **{
                            key: value
                            for key, value in meta.items()
                            if key not in ("inputs", "derived")
                        },
                        "cache_hit": True,
                    }
                except (OSError, ValueError, TypeError, KeyError):
                    pass
                directory = job["directory"]
                state = _action_labels_state(
                    request.action_rules, request.board_roi, request.pose_quality, request.hand_cue
                )
                parent = directory / ".action-labels-cache"
                if not parent.resolve().is_relative_to(directory.resolve()):
                    raise ValueError("Đường dẫn cache nhãn hành động không hợp lệ.")
                parent.mkdir(exist_ok=True)
                generation = uuid.uuid4().hex
                temporary = parent / (".tmp-" + generation)
                temporary.mkdir()
                first_pts, previous_pts, previous_time = None, None, -1.0
                source = manifest["source"]
                base = Fraction(source["time_base"])
                with (
                    (directory / "frames.jsonl").open("rb") as raw,
                    (directory / "frames.idx").open("rb") as source_offsets,
                    (temporary / "frames.jsonl").open("xb") as output,
                    (temporary / "frames.idx").open("xb") as offsets,
                ):
                    for index in range(source["frame_count"]):
                        if job["stop"].exists():
                            raise ValueError("Đã hủy tạo nhãn hành động.")
                        if struct.unpack("<Q", source_offsets.read(8))[0] != raw.tell():
                            raise ValueError("Offset frame nguồn không khớp.")
                        encoded = raw.readline(MAX_FRAME_RANGE_BYTES + 1)
                        if not encoded or len(encoded) > MAX_FRAME_RANGE_BYTES:
                            raise ValueError("Frame nguồn thiếu hoặc vượt giới hạn 4 MiB.")
                        row = json.loads(encoded)
                        if not isinstance(row, dict):
                            raise ValueError("Frame raw phải là object.")
                        json.dumps(row, allow_nan=False)
                        pts, timestamp = row.get("source_pts"), row.get("timestamp_seconds")
                        if (
                            row.get("analysis_task") != "pose"
                            or isinstance(row.get("frame_index"), bool)
                            or row.get("frame_index") != index
                            or isinstance(pts, bool)
                            or not isinstance(pts, int)
                            or row.get("time_base") != source["time_base"]
                            or isinstance(timestamp, bool)
                            or not isinstance(timestamp, (int, float))
                            or not math.isfinite(timestamp)
                            or timestamp <= previous_time
                            or timestamp >= source["duration_seconds"]
                            or previous_pts is not None
                            and pts <= previous_pts
                        ):
                            raise ValueError("Frame/PTS/clock raw Pose không hợp lệ.")
                        first_pts = pts if first_pts is None else first_pts
                        if abs(timestamp - float((pts - first_pts) * base)) > 1e-6:
                            raise ValueError("Timestamp raw không khớp PTS nguồn.")
                        state.update({**row, "width": source["width"], "height": source["height"]})
                        result = {
                            key: row[key]
                            for key in (
                                "frame_index",
                                "source_pts",
                                "time_base",
                                "timestamp_seconds",
                                "segment_id",
                            )
                        }
                        result["action_people"] = state.people
                        payload = (
                            json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
                        ).encode("utf-8")
                        if len(payload) > MAX_FRAME_RANGE_BYTES:
                            raise ValueError("Nhãn hành động vượt giới hạn frame.")
                        offsets.write(struct.pack("<Q", output.tell()))
                        output.write(payload)
                        previous_pts, previous_time = pts, timestamp
                    if raw.read(1) or source_offsets.read(1):
                        raise ValueError("Ledger raw có frame/index dư.")
                    for stream in (output, offsets):
                        stream.flush()
                        os.fsync(stream.fileno())
                state.finish()
                current, current_manifest, current_namespace = self._action_labels_inputs(job)
                if (
                    current != inputs
                    or current_namespace != namespace
                    or current_manifest != manifest
                ):
                    raise ValueError("Đầu vào thay đổi trong lúc tạo nhãn hành động.")
                meta = dict(
                    schema_version=1,
                    kind="full_frame_action_labels",
                    job_id=job["id"],
                    source_sha256=manifest["source_sha256"],
                    actor_namespace=namespace,
                    generation=generation,
                    frame_count=source["frame_count"],
                    raw_frames_sha256=inputs["frames.jsonl"]["sha256"],
                    rule_config=state.config,
                    rule_implementation_sha256=ACTION_RULE_IMPLEMENTATION_SHA256,
                    rule_implementation_sources=dict(ACTION_RULE_IMPLEMENTATION_SOURCES),
                    integrity_scope=ACTION_LABEL_INTEGRITY_SCOPE,
                    action_rules=request.action_rules,
                    pose_quality=request.pose_quality,
                    hand_cue=request.hand_cue,
                    board_roi=request.board_roi,
                    board_roi_source_sha256=request.board_roi_source_sha256
                    if request.board_roi is not None
                    else None,
                    ground_truth=False,
                    scope="causal_saved_raw_pose_observations",
                    inputs=inputs,
                    derived={
                        name: _action_file_identity(
                            temporary / name, job.setdefault("_action_label_fingerprints", {})
                        )
                        for name in ("frames.jsonl", "frames.idx")
                    },
                )
                temporary.rename(parent / generation)
                temporary = None
                _write_json(directory / "action-labels.meta.json", meta)
                return {
                    **{
                        key: value
                        for key, value in meta.items()
                        if key not in ("inputs", "derived")
                    },
                    "cache_hit": False,
                }
            except HTTPException:
                raise
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                struct.error,
                ZeroDivisionError,
            ) as exc:
                raise HTTPException(409, "Chưa thể chuẩn bị nhãn hành động: " + str(exc)) from exc
            finally:
                if temporary is not None and temporary.exists():
                    shutil.rmtree(temporary)

    def action_labels(self, job, start, count, generation=None):
        with self._action_labels_lock(job):
            try:
                inputs, manifest, namespace = self._action_labels_inputs(job)
                meta, cache = self._action_labels_cached(job, inputs, manifest, namespace)
                if generation is not None and generation != meta["generation"]:
                    raise HTTPException(
                        409, "Thế hệ nhãn hành động đã thay đổi; cần chuẩn bị lại bản đang xem."
                    )
                if start >= meta["frame_count"]:
                    raise HTTPException(404, "Frame ngoài phạm vi nhãn hành động.")
                count = min(count, meta["frame_count"] - start)
                rows, size = [], 0
                with (
                    (cache / "frames.idx").open("rb") as offsets,
                    (cache / "frames.jsonl").open("rb") as frames,
                ):
                    offsets.seek(start * 8)
                    for index in range(start, start + count):
                        frames.seek(struct.unpack("<Q", offsets.read(8))[0])
                        encoded = frames.readline(MAX_FRAME_RANGE_BYTES - size + 1)
                        size += len(encoded)
                        if size > MAX_FRAME_RANGE_BYTES:
                            raise HTTPException(
                                413, "Nhãn hành động vượt giới hạn 4 MiB; giảm count."
                            )
                        row = json.loads(encoded)
                        if row["frame_index"] != index:
                            raise ValueError("Index cache nhãn hành động không khớp.")
                        rows.append(row)
                return {
                    **{
                        key: value
                        for key, value in meta.items()
                        if key not in ("inputs", "derived")
                    },
                    "start": start,
                    "count": count,
                    "frames": rows,
                }
            except HTTPException:
                raise
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                struct.error,
                ZeroDivisionError,
            ) as exc:
                raise HTTPException(
                    409, "Nhãn hành động chưa chuẩn bị hoặc cache đã cũ/hỏng: " + str(exc)
                ) from exc

    def context(self, job, reviewer_id=None):
        if reviewer_id is not None and (
            not isinstance(reviewer_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", reviewer_id) is None
        ):
            raise HTTPException(
                422, "Mã nhánh người duyệt dùng 1–40 chữ/số ASCII, gạch dưới hoặc gạch nối."
            )
        if job["state"] != "complete":
            raise HTTPException(409, "Chỉ gán nhãn khi lượt video đã hoàn tất.")
        self.require_preview(job)
        duration = job["source_meta"]["duration_seconds"]
        if "track_segments" not in job or "scene_segments" not in job:
            segments, latest, scenes = [], {}, []
            with (job["directory"] / "frames.jsonl").open(encoding="utf-8") as stream:
                rows = (json.loads(line) for line in stream)
                row = next(rows, None)
                while row is not None:
                    following = next(rows, None)
                    end = min(duration, following["timestamp_seconds"] if following else duration)
                    scene_id = row["segment_id"] if row["segment_id"] is not None else "source"
                    start = row["timestamp_seconds"]
                    if end > start:
                        if (
                            scenes
                            and scenes[-1]["segment_id"] == scene_id
                            and abs(scenes[-1]["end_s"] - start) <= 1e-8
                        ):
                            scenes[-1]["end_s"] = end
                        else:
                            scenes.append({"segment_id": scene_id, "start_s": start, "end_s": end})
                    for track in row["tracks"]:
                        if not track.get("confirmed"):
                            continue
                        key = (row["segment_id"], track["track_id"])
                        start = row["timestamp_seconds"]
                        if key not in latest or abs(latest[key]["end_s"] - start) > 1e-8:
                            extent = {
                                "segment_id": key[0],
                                "track_id": key[1],
                                "start_s": start,
                                "end_s": end,
                            }
                            if end > start:
                                segments.append(extent)
                                latest[key] = extent
                        else:
                            latest[key]["end_s"] = end
                    row = following
            job["track_segments"] = segments
            job["scene_segments"] = scenes
        saved = sorted(
            (p for p in job["directory"].glob("review-*.json") if REVIEW_NAME.fullmatch(p.name)),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        reviewed, warnings = {}, []
        manifest = _json(job["directory"] / "manifest.json")
        source_sha = manifest["source_sha256"]
        namespace = (
            "run:" + str(manifest["run_id"]) if manifest.get("run_id") else "job:" + job["id"]
        )
        review_args = dict(
            duration_s=duration,
            model_segments=job["track_segments"],
            scene_segments=job["scene_segments"],
            source_sha256=source_sha,
            actor_namespace=namespace,
            require_actor_references=job.get("processing_mode") == "annotation_prep",
        )
        for path in saved:
            try:
                candidate = _json(path)
                json.dumps(candidate, allow_nan=False)
                if not isinstance(candidate, dict):
                    raise ValueError("snapshot must be an object")
                if reviewer_id is not None:
                    metadata = candidate.get("review_metadata", {})
                    if not isinstance(metadata, dict) or metadata.get("reviewer_id") != reviewer_id:
                        continue
                if candidate["schema_version"] != 2 or candidate["source_sha256"] != source_sha:
                    raise ValueError("snapshot source/schema mismatch")
                validated = validate_review(candidate, **review_args)
                self.validate_actor_reference_frames(job, validated["human_actor_references"])
                reviewed = {
                    **validated,
                    "review_snapshot": path.name,
                }
                break
            except (OSError, ValueError, TypeError, KeyError) as exc:
                warnings.append(f"Bỏ qua bản nhãn không đọc/kiểm tra được {path.name}: {exc}")
        if warnings:
            warnings.append(
                "Đã khôi phục bản nhãn hợp lệ trước đó."
                if reviewed
                else "Chưa khôi phục được bản nhãn nào; các bản lỗi được giữ nguyên trên đĩa, cần kiểm tra/khôi phục thủ công."
            )
        empty = {"events": [], "coverage": []}
        if reviewer_id is not None:
            empty["review_metadata"] = {"reviewer_id": reviewer_id, "adjudication_state": "draft"}
        canonical = validate_review(reviewed or empty, **review_args)
        return {
            **canonical,
            "review_branch": reviewer_id,
            "source_kind": manifest.get("kind"),
            "source_sha256": source_sha,
            "review_snapshot": reviewed.get("review_snapshot"),
            "duration_s": job["source_meta"]["duration_seconds"],
            "model_track_segments": job["track_segments"],
            "scene_segments": job["scene_segments"],
            "coverage": reviewed.get("coverage", []),
            "events": reviewed.get("events", []),
            "warnings": warnings,
        }

    def validate_actor_reference_frames(self, job, references):
        if not references:
            return
        count = job["source_meta"]["frame_count"]
        directory = job["directory"]
        with (
            (directory / "frames.idx").open("rb") as offsets,
            (directory / "frames.jsonl").open("rb") as frames,
        ):
            for reference in references:
                index = reference["frame_index"]
                if not 0 <= index < count:
                    raise ValueError("Ảnh tham chiếu nằm ngoài timeline nguồn.")
                offsets.seek(index * 8)
                packed = offsets.read(8)
                if len(packed) != 8:
                    raise ValueError("Thiếu chỉ mục ảnh tham chiếu nguồn.")
                frames.seek(struct.unpack("<Q", packed)[0])
                body = frames.readline(MAX_FRAME_RANGE_BYTES + 1)
                if len(body) > MAX_FRAME_RANGE_BYTES:
                    raise ValueError("Frame tham chiếu vượt giới hạn đọc.")
                row = json.loads(body)
                segment = row["segment_id"] if row["segment_id"] is not None else "source"
                if (
                    row["frame_index"] != index
                    or reference["segment_id"] != segment
                    or reference["source_pts"] != row["source_pts"]
                    or Fraction(reference["time_base"]) != Fraction(row["time_base"])
                    or abs(reference["timestamp_seconds"] - row["timestamp_seconds"]) > 1e-9
                ):
                    raise ValueError(
                        "Mốc tham chiếu không khớp frame/PTS/time base/thời gian nguồn."
                    )

    def pilot_context(self, job, reviewer_id=None):
        context = self.context(job, reviewer_id=reviewer_id)
        result = {
            "job_id": job["id"],
            **{
                k: context[k]
                for k in ("review_branch", "source_sha256", "actor_namespace", "review_snapshot")
            },
            "pilot": None,
            "audit": None,
            "warnings": list(context["warnings"]),
        }
        found = []
        for index, path in enumerate(self.directory.glob("pilot_*/pilot_manifest.json")):
            if index >= MAX_PILOT_MANIFESTS:
                result["warnings"].append(
                    "Có quá nhiều manifest pilot; cần kiểm tra danh sách nguồn."
                )
                return result
            try:
                if not path.resolve().is_relative_to(self.directory.resolve()):
                    raise ValueError("pilot outside workspace")
                with path.open("rb") as stream:
                    encoded = stream.read(MAX_PILOT_MANIFEST_BYTES + 1)
                if len(encoded) > MAX_PILOT_MANIFEST_BYTES:
                    raise ValueError("pilot manifest size limit")
                value = json.loads(encoded)
                if not isinstance(value, dict):
                    raise ValueError("pilot manifest must be an object")
                json.dumps(value, allow_nan=False)
                if value.get("source_sha256") == context["source_sha256"]:
                    found.append((path, value))
            except (OSError, ValueError, TypeError):
                result["warnings"].append(
                    "Một manifest pilot không đọc được hoặc nằm ngoài thư mục hợp lệ."
                )
        if not found:
            return result
        if len(found) != 1:
            result["warnings"].append(
                "Có nhiều manifest pilot cho cùng nguồn; chưa tự chọn một bản."
            )
            return result
        path, value = found[0]
        try:
            pilot = self.verified_pilot(job, value, path.parent.name)
            audit = audit_pilot_review(context, pilot["windows"], duration_s=context["duration_s"])
        except (OSError, ValueError, KeyError, TypeError, struct.error):
            result["warnings"].append(
                "Cửa sổ pilot chưa khớp nguồn, frame hoặc PTS đã xác minh; chưa nạp danh sách."
            )
            return result
        result.update(pilot=pilot, audit=audit)
        declared_split = context["review_metadata"].get("split")
        if declared_split and declared_split != pilot["approved_source_split"]:
            result["warnings"].append(
                "Split trong bản nhãn khác split nguồn pilot; cần người duyệt kiểm tra."
            )
        return result

    def verified_pilot(self, job, value, pilot_id):
        source, declared = job["source_meta"], value["source"]
        if value.get("schema_version") != 2 or not isinstance(declared, dict):
            raise ValueError("unsupported pilot manifest")
        for key in ("width", "height", "frame_count", "time_base", "duration_seconds"):
            if source.get(key) != declared.get(key) or source.get(key) is None:
                raise ValueError("pilot source metadata mismatch")
        split = value.get("approved_source_split")
        if split not in ("train", "val", "test", "unassigned"):
            raise ValueError("pilot source split is unknown")
        candidates = value.get("candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_PILOT_WINDOWS:
            raise ValueError("pilot window limit")
        directory = job["directory"]
        index_path = directory / "frames.idx"
        if index_path.stat().st_size != source["frame_count"] * 8:
            raise ValueError("source frame index count mismatch")
        windows = []
        with index_path.open("rb") as offsets, (directory / "frames.jsonl").open("rb") as frames:

            def frame_at(index):
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < source["frame_count"]
                ):
                    raise ValueError("pilot frame index invalid")
                offsets.seek(index * 8)
                frames.seek(struct.unpack("<Q", offsets.read(8))[0])
                encoded = frames.readline(MAX_FRAME_RANGE_BYTES + 1)
                if len(encoded) > MAX_FRAME_RANGE_BYTES:
                    raise ValueError("pilot source row size limit")
                row = json.loads(encoded)
                if (
                    not isinstance(row, dict)
                    or row.get("frame_index") != index
                    or row.get("time_base") != source["time_base"]
                ):
                    raise ValueError("pilot source frame identity mismatch")
                return row

            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError("pilot window must be object")
                first, last = candidate["source_first_frame"], candidate["source_last_frame"]
                start_row, end_row = frame_at(first), frame_at(last)
                count = candidate.get("frame_count")
                if first > last or isinstance(count, bool) or count != last - first + 1:
                    raise ValueError("pilot source frame range mismatch")
                for key, row in (("source_start_pts", start_row), ("source_last_pts", end_row)):
                    pts = candidate.get(key)
                    if (
                        isinstance(pts, bool)
                        or not isinstance(pts, int)
                        or pts != row["source_pts"]
                    ):
                        raise ValueError("pilot native PTS mismatch")
                end = (
                    frame_at(last + 1)["timestamp_seconds"]
                    if last + 1 < source["frame_count"]
                    else source["duration_seconds"]
                )
                for key, expected in (("start_s", start_row["timestamp_seconds"]), ("end_s", end)):
                    actual = candidate.get(key)
                    if (
                        isinstance(actual, bool)
                        or not isinstance(actual, (float, int))
                        or not math.isfinite(actual)
                        or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8)
                    ):
                        raise ValueError("pilot source timestamp mismatch")
                windows.append(
                    {
                        "window_id": candidate["candidate_id"],
                        **{
                            k: candidate[k]
                            for k in (
                                "start_s",
                                "end_s",
                                "source_first_frame",
                                "source_last_frame",
                                "source_start_pts",
                                "source_last_pts",
                            )
                        },
                        "time_base": source["time_base"],
                    }
                )
        return {
            "pilot_id": pilot_id,
            "source_duration_seconds": source["duration_seconds"],
            "approved_source_split": split,
            "windows": windows,
        }

    def class_summary(self, job, window_seconds, reviewer_id=None):
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise HTTPException(422, "Cửa sổ phải là số giây hữu hạn lớn hơn 0.")
        context = self.context(job, reviewer_id=reviewer_id)
        duration = context["duration_s"]
        if duration / window_seconds > MAX_CLASS_WINDOWS:
            raise HTTPException(
                422, f"Chọn cửa sổ lớn hơn để có tối đa {MAX_CLASS_WINDOWS} khoảng."
            )
        events = context["events"]
        summary = aggregate_events(
            events,
            duration_s=duration,
            coverage=context["coverage"],
            window_s=window_seconds,
        )

        def saved_intervals(start, end):
            # These are saved annotation spans intersecting this window, not
            # observed behavior onsets. Counts can repeat across windows.
            active = [row for row in events if row["start_s"] < end and row["end_s"] > start]
            return {
                "unknown": sum(row["label"] == "unknown" for row in active),
                "uncertain": sum(row["uncertain"] for row in active),
                "draft": sum(row["review"] != "reviewed" for row in active),
            }

        summary["saved_intervals"] = saved_intervals(0, duration)
        for window in summary["windows"]:
            window["saved_intervals"] = saved_intervals(window["start_s"], window["end_s"])
        return {
            "schema_version": 2,
            **context,
            "window_seconds": window_seconds,
            "class_summary": summary,
        }

    def class_export(self, job, window_seconds, reviewer_id=None):
        """Allowlisted class aggregates for a local consumer; no actor or free-text fields."""
        context = self.class_summary(job, window_seconds, reviewer_id=reviewer_id)
        summary = context["class_summary"]
        metric_keys = (
            "count",
            "truncated_episode_count",
            "duration_s",
            "observed_duration_s",
            "reviewed_coverage_actor_s",
        )

        def labels(values):
            return {label: {key: values[label][key] for key in metric_keys} for label in LABELS}

        def uncertainty(values):
            return {key: values[key] for key in ("unknown", "uncertain", "draft")}

        return {
            "schema_version": 1,
            "kind": "class_observation_summary",
            "source_ref": {"sha256": context["source_sha256"]},
            "source_clock": {
                "duration_s": context["duration_s"],
                "window_seconds": context["window_seconds"],
                "interval": "[start_s,end_s)",
            },
            "saved_data_only": True,
            "saved_snapshot_available": context["review_snapshot"] is not None,
            "class_session": labels(summary["session"]),
            "windows": [
                {
                    "start_s": row["start_s"],
                    "end_s": row["end_s"],
                    "labels": labels(row["labels"]),
                    "uncertainty_counts": uncertainty(row["saved_intervals"]),
                }
                for row in summary["windows"]
            ],
            "uncertainty_counts": uncertainty(summary["saved_intervals"]),
            "excluded_event_count": summary["excluded_events"],
            "caveats": [
                summary["caveat"],
                "Saved review data only; browser drafts are excluded. This aggregate is not GT or an interaction rating.",
                "Uncertainty counts count saved annotation spans; categories may overlap. Missing review coverage is not a negative label.",
                "Source SHA256 is linkable to a known private video. Omitting actor fields is not full anonymization.",
            ],
        }

    def close(self):
        with self.lock:
            self.closed = True
            for job in self.jobs.values():
                if job["state"] not in TERMINAL:
                    job["stop"].touch()
                self.cancel_video_export(job)
        self.executor.shutdown(wait=True, cancel_futures=True)


def create_app(root=PROJECT_ROOT, *, runner=run_video, replay_runner=None):
    root = Path(root).resolve()
    jobs = DemoJobs(root, runner, replay_runner)

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.close()

    app = FastAPI(title="Classroom Video Lab", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "::1", "testserver"]
    )
    app.state.jobs, app.state.token = jobs, secrets.token_urlsafe(32)
    app.state.clock_id = uuid.uuid4().hex
    app.state.max_upload_bytes = 2 * 1024**3

    @app.middleware("http")
    async def local_guard(request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if (
                origin and urlparse(origin).netloc != request.headers.get("host")
            ) or not secrets.compare_digest(
                request.headers.get("x-demo-token", ""), app.state.token
            ):
                return JSONResponse(
                    {"detail": "Yêu cầu thay đổi phải đến từ web demo local."}, status_code=403
                )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/", response_class=HTMLResponse)
    def index():
        return Path(__file__).with_name("web.html").read_text(encoding="utf-8")

    @app.get("/api/clock")
    def clock():
        return {"monotonic_seconds": time.monotonic(), "clock_id": app.state.clock_id}

    @app.get("/api/action-rubric")
    def action_rubric():
        import yaml

        path = root / "configs" / "action_rubric.yaml"
        if not path.is_file():
            raise HTTPException(404, "Chưa có định nghĩa gán nhãn cho dự án này.")
        if path.stat().st_size > 128 * 1024:
            raise HTTPException(409, "Định nghĩa gán nhãn vượt giới hạn đọc.")
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("version"), str)
            or not isinstance(value.get("labels"), list)
        ):
            raise HTTPException(409, "Định nghĩa gán nhãn chưa đúng cấu trúc.")
        return value

    @app.get("/api/annotation-sources")
    def annotation_sources():
        return jobs.annotation_sources()

    @app.post("/api/annotation-sources", status_code=202)
    def prepare_annotation_source(request: AnnotationSourceInput):
        return jobs.submit_annotation_source(request)

    @app.get("/api/config")
    def settings():
        from classroom_ai.inference.pose import KEYPOINT_NAMES_13, POSE_EDGES_13

        videos = []
        for folder in (root, root / "data" / "videos", jobs.directory / "uploads"):
            if folder.exists():
                for path in sorted(folder.glob("*")):
                    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                        videos.append({"source": str(path.relative_to(root)), "name": path.name})
        return {
            "token": app.state.token,
            "models": [
                {k: v for k, v in model.items() if k != "path"} for model in jobs.models.values()
            ],
            "videos": videos[:200],
            "keypoint_names13": KEYPOINT_NAMES_13,
            "pose_edges13": POSE_EDGES_13,
            "limits": {"active_jobs": 1, "max_upload_bytes": app.state.max_upload_bytes},
        }

    @app.post("/api/uploads", status_code=201)
    async def upload(request: Request, name: str):
        suffix = Path(name).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(422, "Chọn file video MP4/MOV/MKV/AVI/M4V.")
        directory = jobs.directory / "uploads"
        directory.mkdir(exist_ok=True)
        path = directory / (uuid.uuid4().hex + suffix)
        total = 0
        try:
            with path.open("xb") as stream:
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > app.state.max_upload_bytes:
                        raise HTTPException(413, "Video vượt giới hạn dung lượng upload.")
                    stream.write(chunk)
            if not total:
                raise HTTPException(422, "File rỗng.")
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return {"source": str(path.relative_to(root)), "bytes": total}

    @app.post("/api/jobs", status_code=202)
    def start(request: JobInput):
        return jobs.submit(request)

    @app.get("/api/jobs/{job_id}")
    def state(job_id: str):
        return jobs.public(jobs.find(job_id))

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        job = jobs.find(job_id)
        with jobs.lock:
            if job["state"] not in TERMINAL:
                job["stop"].touch()
                if job.get("_replay_stop") is not None:
                    job["_replay_stop"].set()
                jobs.update(
                    job,
                    state="cancelling",
                    message="Đã yêu cầu hủy; chờ thao tác AI hiện tại kết thúc",
                )
            return jobs.public(job)

    @app.get("/api/jobs/{job_id}/latest")
    def latest(job_id: str):
        return {**jobs.latest(jobs.find(job_id)), "clock_id": app.state.clock_id}

    @app.get("/api/jobs/{job_id}/live-frame")
    def live_frame(job_id: str, sequence: int = Query(ge=0)):
        job = jobs.find(job_id)
        jobs.latest(job)
        with jobs.lock:
            pair = job.get("_live_pair")
            if pair is None or pair["sequence"] != sequence:
                raise HTTPException(
                    409, "Mẫu đã thay đổi hoặc chưa có ảnh; đọc latest ở nhịp tiếp theo."
                )
            return Response(
                pair["jpeg"], media_type="image/jpeg", headers={"X-Replay-Sequence": str(sequence)}
            )

    @app.get("/api/jobs/{job_id}/artifacts/{name}")
    def artifact(job_id: str, name: str, inline: bool = False):
        job = jobs.find(job_id)
        allowed = name in ARTIFACTS or REVIEW_NAME.fullmatch(name)
        directory = job.get("directory")
        path = (directory / name).resolve() if directory and allowed else None
        if not path or path.parent != directory.resolve() or not path.is_file():
            raise HTTPException(404, "Không thấy kết quả đã xuất.")
        return FileResponse(
            path,
            filename=name,
            media_type="video/mp4" if name.endswith(".mp4") else None,
            # Inline playback applies only to explicitly allowlisted MP4s.
            # JSON/review artifacts retain attachment disposition.
            content_disposition_type="inline"
            if inline and name in ARTIFACTS and name.endswith(".mp4")
            else "attachment",
        )

    @app.post("/api/jobs/{job_id}/action-labels")
    def prepare_action_labels(job_id: str, request: ActionLabelsInput):
        return jobs.prepare_action_labels(jobs.find(job_id), request)

    @app.post("/api/jobs/{job_id}/video-export", status_code=202)
    def start_video_export(job_id: str, request: VideoExportInput):
        return jobs.start_video_export(jobs.find(job_id), request)

    @app.post("/api/jobs/{job_id}/video-export/cancel")
    def cancel_video_export(job_id: str):
        return jobs.cancel_video_export(jobs.find(job_id))

    @app.get("/api/jobs/{job_id}/video-export/{export_id}/{name}")
    def video_export_artifact(job_id: str, export_id: str, name: str, inline: bool = False):
        job = jobs.find(job_id)
        if (
            not re.fullmatch(r"[a-f0-9]{32}", export_id)
            or name not in {"video.mp4", "report.json"}
            or not job.get("directory")
        ):
            raise HTTPException(404, "Không thấy video đã xuất.")
        latest = job.get("video_export") or {}
        if latest.get("id") == export_id and latest.get("state") != "complete":
            raise HTTPException(404, "Bản xuất chưa hoàn tất.")
        root = job["directory"].resolve()
        directory = (root / "video_exports" / export_id).resolve()
        path = directory / name
        if (
            not directory.is_relative_to(root)
            or not path.is_file()
            or path.resolve().parent != directory
            or any(
                not (directory / auxiliary).is_file()
                or (directory / auxiliary).resolve().parent != directory
                for auxiliary in ("report.json", "web-export.json")
            )
        ):
            raise HTTPException(404, "Không thấy video đã xuất.")
        try:
            if _json(directory / "report.json").get("status") != "completed":
                raise ValueError("incomplete export")
            commit = _json(directory / "web-export.json")
            if commit != {"status": "complete", "job_id": job_id, "export_id": export_id}:
                raise ValueError("uncommitted export")
        except (OSError, ValueError, AttributeError):
            raise HTTPException(404, "Bản xuất chưa được xác nhận hoàn tất.") from None
        return FileResponse(
            path,
            filename=name,
            media_type="video/mp4" if name.endswith(".mp4") else "application/json",
            content_disposition_type="inline" if inline and name == "video.mp4" else "attachment",
        )

    @app.get("/api/jobs/{job_id}/action-labels")
    def action_labels(
        job_id: str,
        start: int = Query(0, ge=0),
        count: int = Query(32, ge=1, le=32),
        generation: str | None = Query(None, pattern=r"^[a-f0-9]{32}$"),
    ):
        return jobs.action_labels(jobs.find(job_id), start, count, generation)

    @app.get("/api/jobs/{job_id}/frames")
    def frame_range(job_id: str, start: int = Query(0, ge=0), count: int = Query(32, ge=1, le=32)):
        job = jobs.find(job_id)
        if job["state"] != "complete":
            raise HTTPException(409, "Kết quả frame chưa sẵn sàng.")
        jobs.require_preview(job)
        directory = job["directory"]
        index_path = directory / "frames.idx"
        total = index_path.stat().st_size // 8
        if start >= total:
            raise HTTPException(404, "Frame ngoài phạm vi kết quả.")
        count = min(count, total - start)
        prefix = f'{{"start":{start},"count":{count},"frames":['.encode()
        rows, size = [], len(prefix) + 2 + count - 1
        with index_path.open("rb") as offsets, (directory / "frames.jsonl").open("rb") as stream:
            offsets.seek(start * 8)
            for _ in range(count):
                stream.seek(struct.unpack("<Q", offsets.read(8))[0])
                # A corrupt/huge JSONL row cannot trigger an unbounded read or response.
                row = stream.readline(MAX_FRAME_RANGE_BYTES - size + 1)
                size += len(row)
                if size > MAX_FRAME_RANGE_BYTES:
                    raise HTTPException(413, "Dữ liệu frame vượt giới hạn 4 MiB; giảm count.")
                rows.append(row.rstrip(b"\r\n"))
        return Response(prefix + b",".join(rows) + b"]}", media_type="application/json")

    @app.get("/api/jobs/{job_id}/frame/{index}")
    def frame(job_id: str, index: int):
        job = jobs.find(job_id)
        if job["state"] != "complete":
            raise HTTPException(409, "Kết quả frame chưa sẵn sàng.")
        jobs.require_preview(job)
        directory = job["directory"]
        index_path = directory / "frames.idx"
        if index < 0 or index >= index_path.stat().st_size // 8:
            raise HTTPException(404, "Frame ngoài phạm vi kết quả.")
        with index_path.open("rb") as offsets, (directory / "frames.jsonl").open("rb") as stream:
            offsets.seek(index * 8)
            stream.seek(struct.unpack("<Q", offsets.read(8))[0])
            return json.loads(stream.readline())

    @app.get("/api/jobs/{job_id}/review-context")
    def context(job_id: str, reviewer_id: str | None = None):
        return jobs.context(jobs.find(job_id), reviewer_id=reviewer_id)

    @app.get("/api/jobs/{job_id}/pilot-context")
    def pilot_context(job_id: str, reviewer_id: str | None = None):
        return jobs.pilot_context(jobs.find(job_id), reviewer_id=reviewer_id)

    @app.get("/api/jobs/{job_id}/class-summary")
    def class_summary(
        job_id: str, window_seconds: float = Query(default=60), reviewer_id: str | None = None
    ):
        return jobs.class_summary(jobs.find(job_id), window_seconds, reviewer_id=reviewer_id)

    @app.get("/api/jobs/{job_id}/class-export")
    def class_export(
        job_id: str, window_seconds: float = Query(default=60), reviewer_id: str | None = None
    ):
        return JSONResponse(
            jobs.class_export(jobs.find(job_id), window_seconds, reviewer_id=reviewer_id),
            headers={"Content-Disposition": 'attachment; filename="class-observations.json"'},
        )

    def save_annotations(job_id: str, payload: dict, reviewer_id: str | None = None):
        job = jobs.find(job_id)
        context = jobs.context(job, reviewer_id=reviewer_id)
        if (
            "expected_snapshot" in payload
            and payload["expected_snapshot"] != context["review_snapshot"]
        ):
            raise HTTPException(409, "Nhánh nhãn đã thay đổi; tải lại trước khi lưu bản nháp.")
        try:
            metadata = payload.get("review_metadata")
            if reviewer_id is not None and (
                not isinstance(metadata, dict) or metadata.get("reviewer_id") != reviewer_id
            ):
                raise ValueError("Người duyệt trong metadata không khớp nhánh đang mở.")
            if not {"events", "coverage"}.issubset(payload):
                raise ValueError("review requires events and coverage")
            reviewed = validate_review(
                payload,
                duration_s=context["duration_s"],
                model_segments=context["model_track_segments"],
                scene_segments=context["scene_segments"],
                source_sha256=context["source_sha256"],
                actor_namespace=context["actor_namespace"],
                require_actor_references=job.get("processing_mode") == "annotation_prep",
            )
            jobs.validate_actor_reference_frames(job, reviewed["human_actor_references"])
            events = reviewed["events"]
            summary = aggregate_events(
                events, duration_s=context["duration_s"], coverage=payload["coverage"]
            )
            bundle = {
                "schema_version": 2,
                **context,
                **reviewed,
                "events": events,
                "coverage": payload["coverage"],
                "source_sha256": _json(job["directory"] / "manifest.json")["source_sha256"],
                "summary": summary,
            }
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        name = "review-" + uuid.uuid4().hex + ".json"
        _publish_review(job["directory"] / name, bundle)
        return {
            **reviewed,
            "review_branch": reviewer_id,
            "snapshot": name,
            "events": events,
            "coverage": bundle["coverage"],
            "summary": summary,
        }

    @app.post("/api/jobs/{job_id}/annotations")
    def annotate(job_id: str, payload: dict, reviewer_id: str | None = None):
        # Read/check/publish are one critical section. A stale importer cannot
        # replace the branch head between its precondition check and publication.
        with jobs.lock:
            return save_annotations(job_id, payload, reviewer_id)

    @app.get("/api/jobs/{job_id}/readiness")
    def readiness(job_id: str, reviewer_id: str | None = None):
        context = jobs.context(jobs.find(job_id), reviewer_id=reviewer_id)
        return {
            key: context[key]
            for key in (
                "evaluation_ready",
                "readiness_reasons",
                "actor_namespace",
                "source_sha256",
                "review_metadata",
                "review_branch",
                "actor_reference_count",
                "actor_reference_missing_span_count",
                "actor_references_required",
                "source_kind",
            )
        }

    @app.get("/api/jobs/{job_id}/gt-bundle")
    def gt_bundle(job_id: str, reviewer_id: str | None = None):
        context = jobs.context(jobs.find(job_id), reviewer_id=reviewer_id)
        if not context["evaluation_ready"]:
            raise HTTPException(
                409,
                {
                    "message": "Chưa đủ điều kiện xuất GT đã duyệt.",
                    "readiness_reasons": context["readiness_reasons"],
                },
            )
        return JSONResponse(
            {"schema_version": 2, "kind": "human_reviewed_events", **context},
            headers={"Content-Disposition": 'attachment; filename="reviewed-gt.json"'},
        )

    @app.post("/api/jobs/{job_id}/proposals")
    def proposals(job_id: str, request: dict | None = None, reviewer_id: str | None = None):
        job = jobs.find(job_id)
        if job.get("processing_mode") == "annotation_prep":
            raise HTTPException(
                409, "Nguồn gán nhãn chưa chạy AI; không tạo gợi ý từ mảng model rỗng."
            )
        context = jobs.context(job, reviewer_id=reviewer_id)
        manifest = _json(job["directory"] / "manifest.json")
        try:
            request = HandRuleInput.model_validate(request or {})
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        raster = manifest.get("source") or job.get("source_meta", {})
        if any(type(raster.get(key)) is not int or raster[key] <= 0 for key in ("width", "height")):
            raise HTTPException(409, "Thiếu kích thước raster nguồn đã xác minh.")

        def rows():
            with (job["directory"] / "frames.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    yield {
                        **json.loads(line),
                        "width": raster["width"],
                        "height": raster["height"],
                    }

        value = {
            "schema_version": 2,
            "kind": "model_prediction",
            "source_sha256": context["source_sha256"],
            "actor_namespace": context["actor_namespace"],
            "weights_sha256": manifest.get("weights_sha256"),
            "model": manifest.get("config", {}).get("weights"),
            "events": propose_raise_hand_events(
                rows(), hand_cue=request.hand_cue, pose_quality=request.pose_quality
            ),
            "hand_cue": request.hand_cue,
            "rule_config": RaiseHandState(
                hand_cue=request.hand_cue, pose_quality=request.pose_quality
            ).config,
            "note": "Luật giơ tay chưa hiệu chỉnh; nháp cần người kiểm, có thể nhầm chạm đầu/vươn vai. Không tự nhận đứng/ngồi/lên bảng/viết bảng.",
        }
        _write_json(job["directory"] / "proposals.json", value)
        return value

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Web demo video local: Detect, ByteTrack, Pose và timeline"
    )
    parser.add_argument("--port", type=int, default=8090)
    arguments = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=arguments.port, workers=1)


if __name__ == "__main__":
    main()
