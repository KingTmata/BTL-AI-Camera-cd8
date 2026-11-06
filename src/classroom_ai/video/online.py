"""Causal, PTS-paced replay of a recorded local source, not a live camera.

Frame callbacks run on the inference thread and must return promptly. Terminal
``on_event`` runs after that worker joins; no frame callback accompanies it. ``on_frame``
receives JPEG bytes and the matching update (same sequence/source frame). Times
end at callback entry, not network receipt or browser presentation. Native
model calls cancel cooperatively when they return; pending work is bounded to1.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from classroom_ai.paths import PROJECT_ROOT
from classroom_ai.video.actions import RaiseHandState
from classroom_ai.video.core import LatestMailbox, TimeSampler, VideoConfig
from classroom_ai.video.media import MediaCancelled, decoder_info, frames, probe_source
from classroom_ai.video.metrics import Metrics
from classroom_ai.video.pose_rules import BasicPoseActionState, validate_board_roi
from classroom_ai.video.tracking import PersonTracker, SceneCuts


@dataclass
class ReplayConfig:
    source: Path
    weights: Path
    device: str = "intel:gpu"
    imgsz: int = 1280
    confidence: float = 0.05
    sample_fps: float = 5.0
    tracking: str = "bytetrack"
    nms_iou: float | None = None
    max_lost_seconds: float = 3.0
    track_high_thresh: float | None = None
    track_low_thresh: float | None = None
    new_track_thresh: float | None = None
    recovery_confirm_observations: int = 2
    output_dir: Path = PROJECT_ROOT / "runs/replay"
    analysis_task: str = "pose"
    decoder: str = "pyav"
    action_rules: str = "off"
    board_roi: list[float] | None = None
    board_roi_source_sha256: str | None = None
    pose_quality: dict | None = None
    hand_cue: str = "nose_both_arms_v1"

    def validate(self):
        if self.analysis_task != "pose":
            raise ValueError("Recorded-stream replay hiện chỉ hỗ trợ Pose.")
        if self.decoder not in ("opencv", "pyav"):
            raise ValueError("decoder phải là opencv hoặc pyav.")
        if self.action_rules not in ("off", "raise_hand", "basic_pose"):
            raise ValueError("action_rules phải là off, raise_hand hoặc basic_pose.")
        if self.action_rules != "off" and self.tracking != "bytetrack":
            raise ValueError("action_rules cần bytetrack để có actor quan sát được.")
        validate_board_roi(self.board_roi)
        from classroom_ai.video.pose_quality import PoseQualityGate

        PoseQualityGate(self.pose_quality)
        if self.hand_cue not in ("nose_both_arms_v1", "shoulder_any_arm_v1"):
            raise ValueError("unsupported hand_cue")
        if self.board_roi is not None and (
            not isinstance(self.board_roi_source_sha256, str)
            or re.fullmatch(r"[a-f0-9]{64}", self.board_roi_source_sha256) is None
        ):
            raise ValueError("board_roi_source_sha256 phải gắn vùng bảng với đúng nguồn.")
        VideoConfig(
            Path(self.source),
            "full",
            weights=Path(self.weights),
            device=self.device,
            analysis_task="pose",
            imgsz=self.imgsz,
            confidence=self.confidence,
            sample_fps=self.sample_fps,
            tracking=self.tracking,
            nms_iou=self.nms_iou,
            track_max_lost_seconds=self.max_lost_seconds,
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            recovery_confirm_observations=self.recovery_confirm_observations,
        ).validate()


def _json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )


def _line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def _check_stop(stop):
    if stop.is_set():
        raise InterruptedError("Đã hủy recorded-stream replay.")


def _hash(path, stop):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            _check_stop(stop)
            digest.update(block)
    return digest.hexdigest()


def _file_state(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _lifetime_latency(path):
    # Disk is the lifetime ledger; only scalar timings are loaded at finalization,
    # never a retained frame/pose history. Active replay memory stays bounded.
    values = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            values.append(json.loads(line)["source_to_publish_ms"])
    return {
        "count": len(values),
        "p50_ms": float(np.percentile(values, 50)) if values else None,
        "p95_ms": float(np.percentile(values, 95)) if values else None,
        "max_ms": max(values) if values else None,
        "over_3s": sum(value >= 3000 for value in values),
        "scope": "All published samples, scheduled 1x source release to server callback entry; excludes browser presentation.",
    }


def run_replay(
    config,
    *,
    on_update,
    on_frame=None,
    on_event=None,
    stop=None,
    predictor_factory=None,
    clock=None,
    wait=None,
    manager=None,
):
    """Return an artifact directory with completed/cancelled/failed report.

    Preflight/full-source PTS validation, hashes, model load/export/compile and
    warmup occur before the 1x source clock starts and remain in cold timings.
    No completed MP4 or frame index is needed to consume updates. Tracking is
    causal; source indices/PTS and separate observation indices are both kept.
    Optional raise_hand/basic_pose emit uncalibrated rule/draft proposals, never GT.
    on_event receives durable envelopes after frame publication or terminal
    flush, which has no synthetic frame/JPEG/release. Callback cost is separate.
    """
    config.validate()
    clock = clock or time.monotonic
    wait = wait or (lambda event, delay: event.wait(delay))
    stop = stop if stop is not None else threading.Event()
    directory = Path(config.output_dir).resolve() / ("replay_" + uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    updates_path, latency_path = directory / "updates.jsonl", directory / "latencies.jsonl"
    updates_path.touch()
    latency_path.touch()
    actions_path = directory / "actions.jsonl"
    actions_path.touch()
    started = clock()
    metrics = Metrics()
    action_state = (
        BasicPoseActionState(
            config.board_roi, pose_quality=config.pose_quality, hand_cue=config.hand_cue
        )
        if config.action_rules == "basic_pose"
        else RaiseHandState(pose_quality=config.pose_quality, hand_cue=config.hand_cue)
        if config.action_rules == "raise_hand"
        else None
    )
    action_status = config.action_rules + "_rule_draft" if action_state is not None else "not_run"
    action_rules = {
        "mode": config.action_rules,
        "engine": "basic_pose_v1"
        if config.action_rules == "basic_pose"
        else "incremental_raise_hand_v1"
        if action_state is not None
        else None,
        "config": action_state.config if action_state is not None else None,
        "input": "raw COCO17, matched confirmed observations only; no imputed feet",
        "status": action_status,
        "ground_truth": False,
        "coverage": "not_reviewed",
        "supported_labels": ["raise_hand", "stand_up", "move_to_board"]
        if config.action_rules == "basic_pose"
        else ["raise_hand"]
        if action_state is not None
        else [],
        "limitations": ["uncalibrated_pose_proxy_not_reviewed_coverage"]
        if action_state is not None
        else [],
        "availability_scope": "Rule result computed before durable write/frame publication/event callback; not consumer or browser delivery.",
    }
    action_summary = dict(
        emitted_draft_events=0, terminal_flush_events=0, callback_count=0, callback_ms=0.0
    )
    last_action_observation = None
    manifest = {
        "schema_version": 2,
        "mode": "recorded_stream_replay_1x",
        "analysis_task": "pose",
        "run_id": directory.name,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "action_status": action_status,
        "action_rules": action_rules,
        "identity_scope": "run/scene-local tracks only",
        "clock": "relative source PTS mapped to server monotonic clock after readiness",
        "timing_scope": "source release to server publication callback entry; no web presentation measurement",
        "source_clock_origin_monotonic": None,
        "requested_device": config.device,
        "resolved_device": None,
    }
    counts = dict(
        frames_released=0,
        frames_selected=0,
        frames_published=0,
        observations_without_detections=0,
        observation_gaps=0,
    )
    first_publication = {"warm_ms": None, "cold_ms": None}
    preparation = {}
    errors = []
    origin = None
    effective_fps = None
    mailbox = LatestMailbox()
    worker = None
    tracker = None
    status = "failed"

    def record_events(events, observation, *, terminal=False):
        if not events:
            return []
        envelopes = []
        with actions_path.open("a", encoding="utf-8") as ledger:
            for event in events:
                available = clock()
                released = None if terminal else observation["source_release_monotonic"]
                envelope = {
                    "schema_version": 1,
                    "event_id": f"{directory.name}:{action_summary['emitted_draft_events']}",
                    "event": event,
                    "availability": {
                        "reason": event["evidence"]["boundary_reason"]["end"],
                        "source_frame_index": observation["source_frame_index"],
                        "source_pts": observation["source_pts"],
                        "time_base": observation["time_base"],
                        "observed_through_s": observation["timestamp_seconds"],
                        "available_monotonic": available,
                        "source_release_monotonic": released,
                        "source_to_available_ms": None
                        if released is None
                        else max(0, available - released) * 1000,
                    },
                    "observation_history_s": event["end_s"] - event["start_s"],
                    "decision_history_s": observation["timestamp_seconds"] - event["start_s"],
                }
                _line(ledger, envelope)
                action_summary["emitted_draft_events"] += 1
                action_summary["terminal_flush_events"] += int(terminal)
                envelopes.append(envelope)
        return envelopes

    def deliver_events(envelopes):
        if on_event is not None:
            for envelope in envelopes:
                stage = clock()
                try:
                    on_event(envelope)
                finally:
                    action_summary["callback_count"] += 1
                    action_summary["callback_ms"] += max(0, clock() - stage) * 1000

    try:
        _check_stop(stop)
        stage = clock()
        manifest["decoder"] = decoder_info(config.decoder)
        preparation["decoder_setup_ms"] = (clock() - stage) * 1000
        inputs = {
            Path(config.source): _file_state(Path(config.source)),
            Path(config.weights): _file_state(Path(config.weights)),
        }

        def verify_inputs():
            if any(_file_state(path) != expected for path, expected in inputs.items()):
                raise ValueError(
                    "Nguồn video/weights thay đổi trong khi replay; provenance không còn hợp lệ."
                )

        stage = clock()
        source = probe_source(Path(config.source), allow_vfr=True, cancel=stop.is_set)
        preparation["source_preflight_ms"] = (clock() - stage) * 1000
        if source.width * source.height * 3 > 128 * 1024 * 1024:
            raise ValueError("Một frame vượt giới hạn 128 MiB.")
        effective_fps = min(config.sample_fps, float(source.fps))
        if action_state is not None:
            max_gap = action_state.config.get("max_gap_s", 0.25)
            sufficient = 1 / effective_fps <= max_gap + 1e-9
            action_rules["cadence"] = {
                "effective_nominal_sample_fps": effective_fps,
                "max_gap_s": max_gap,
                "nominally_sufficient": sufficient,
                "note": "Actual source/sampling/overload gaps still reset the rule; zero drafts never asserts absence.",
            }
            if not sufficient:
                action_rules["limitations"].append("nominal_cadence_below_rule_max_gap")
        manifest["effective_nominal_sample_fps"] = effective_fps
        manifest["source"] = source.metadata()
        stage = clock()
        manifest["source_sha256"] = _hash(Path(config.source), stop)
        if (
            config.board_roi is not None
            and config.board_roi_source_sha256 != manifest["source_sha256"]
        ):
            raise ValueError("board_roi source SHA256 không khớp video; dừng trước inference.")
        manifest["weights_sha256"] = _hash(Path(config.weights), stop)
        preparation["source_and_weights_hash_ms"] = (clock() - stage) * 1000
        verify_inputs()
        _check_stop(stop)
        from classroom_ai.inference.devices import cuda_devices, validate_device

        stage = clock()
        if config.device == "auto":
            cuda, reason = cuda_devices()
            device = next(iter(cuda), "cpu")
            manifest["device_selection_reason"] = (
                reason or "Available CUDA probe selected first device."
            )
        else:
            device = validate_device(config.device)
        manifest["resolved_device"] = device
        preparation["device_probe_ms"] = (clock() - stage) * 1000
        if predictor_factory is None:
            from classroom_ai.inference.pose import PoseInspector
            from classroom_ai.training.manager import RunManager

            predictor_factory = PoseInspector
            if manager is None:
                manager = RunManager(PROJECT_ROOT)

        def slot():
            return manager.inference_slot() if manager is not None else nullcontext()

        stage = clock()
        with slot():
            _check_stop(stop)
            predictor = predictor_factory(Path(config.weights).resolve(), device)
        if hasattr(predictor, "set_cancel"):
            predictor.set_cancel(stop.is_set)
        preparation["model_load_ms"] = (clock() - stage) * 1000
        stage = clock()

        def predict(image):
            with slot():
                _check_stop(stop)
                return predictor.predict(
                    image,
                    config.confidence,
                    imgsz=config.imgsz,
                    extra_nms_iou=config.nms_iou,
                    passes=1,
                )

        warm = predict(np.zeros((source.height, source.width, 3), dtype=np.uint8))
        preparation["model_warmup_export_compile_ms"] = (clock() - stage) * 1000
        manifest["runtime"] = warm.get("runtime", {})
        if (
            warm.get("runtime", {}).get("source_sha256", manifest["weights_sha256"])
            != manifest["weights_sha256"]
        ):
            raise ValueError("Model runtime khác SHA256 weights gốc; dừng replay.")
        _check_stop(stop)
        if config.tracking == "bytetrack":
            tracker = PersonTracker(
                effective_fps,
                config.confidence,
                config.max_lost_seconds,
                metrics,
                max_observation_gap_s=1.5 / effective_fps,
                track_high_thresh=config.track_high_thresh,
                track_low_thresh=config.track_low_thresh,
                new_track_thresh=config.new_track_thresh,
                recovery_confirm_observations=config.recovery_confirm_observations,
            )
        cuts = SceneCuts()
        origin = clock()
        manifest["source_clock_origin_monotonic"] = origin
        preparation["model_ready_cold_ms"] = (origin - started) * 1000
        _json(directory / "manifest.json", manifest)

        def inference():
            nonlocal last_action_observation
            previous_time, previous_index = None, None
            try:
                with (
                    updates_path.open("a", encoding="utf-8") as output,
                    latency_path.open("a", encoding="utf-8") as latencies,
                ):
                    sequence = 0
                    while not stop.is_set():
                        task = mailbox.take(stop)
                        if task is None:
                            break
                        packet, due, queued = task
                        infer_started = clock()
                        _check_stop(stop)
                        verify_inputs()
                        result = predict(packet.image)
                        _check_stop(stop)
                        inferred = clock()
                        if tracker is not None:
                            if cuts.update(packet.image):
                                tracker.reset()
                            track_row = tracker.update(
                                packet.frame_index,
                                packet.timestamp_seconds,
                                result["detections"],
                                observation_index=sequence,
                            )
                        else:
                            track_row = {"segment_id": 0, "tracks": []}
                        # PersonTracker emits only matches for the current native
                        # detection. No display smoothing or lost-track guesses.
                        track_row["tracks"] = [
                            dict(
                                track,
                                state="observed"
                                if track.get("confirmed") is True
                                and isinstance(track.get("detection_index"), int)
                                and not isinstance(track.get("detection_index"), bool)
                                and 0 <= track["detection_index"] < len(result["detections"])
                                else "unknown",
                            )
                            for track in track_row["tracks"]
                        ]
                        row = {
                            "schema_version": 2,
                            "mode": "recorded_stream_replay_1x",
                            "analysis_task": "pose",
                            "sequence": sequence,
                            "observation_index": sequence,
                            "frame_index": packet.frame_index,
                            "source_frame_index": packet.frame_index,
                            "source_pts": packet.source_pts,
                            "time_base": packet.time_base,
                            "timestamp_seconds": packet.timestamp_seconds,
                            "source_release_monotonic": due,
                            "enqueued_monotonic": queued,
                            "infer_started_monotonic": infer_started,
                            "processing_ms": result["processing_ms"],
                            "detections": result["detections"],
                            "segment_id": track_row["segment_id"],
                            "tracks": track_row["tracks"],
                            "observation": "model_prediction",
                            "action_status": action_status,
                            "width": source.width,
                            "height": source.height,
                            "coverage": "sampled_model_observations_not_label_review",
                            "source_frame_gap": None
                            if previous_index is None
                            else packet.frame_index - previous_index - 1,
                            "observation_gap_seconds": None
                            if previous_time is None
                            else packet.timestamp_seconds - previous_time,
                        }
                        last_action_observation = row
                        row["action_events"] = (
                            record_events(action_state.update(row), row)
                            if action_state is not None
                            else []
                        )
                        row["action_state"] = (
                            action_state.counts if action_state is not None else None
                        )
                        row["action_people"] = (
                            action_state.people if action_state is not None else []
                        )
                        jpeg = None
                        if on_frame is not None:
                            ok, encoded = cv2.imencode(
                                ".jpg", packet.image, [cv2.IMWRITE_JPEG_QUALITY, 85]
                            )
                            if not ok:
                                raise RuntimeError("Không mã hóa được JPEG của frame replay.")
                            jpeg = encoded.tobytes()
                        _check_stop(stop)
                        published = clock()
                        row["publish_monotonic"] = published
                        timing = {
                            "sequence": sequence,
                            "frame_index": packet.frame_index,
                            "source_pts": packet.source_pts,
                            "timestamp_seconds": packet.timestamp_seconds,
                            "release_lateness_ms": max(0, queued - due) * 1000,
                            "queue_wait_ms": max(0, infer_started - queued) * 1000,
                            "inference_callback_ms": (inferred - infer_started) * 1000,
                            "source_to_publish_ms": max(0, published - due) * 1000,
                        }
                        row["timing"] = timing
                        row["late_over_3s"] = timing["source_to_publish_ms"] >= 3000
                        if (
                            previous_time is not None
                            and packet.timestamp_seconds - previous_time
                            > 1.5 / effective_fps + 1e-9
                        ):
                            counts["observation_gaps"] += 1
                            row["unknown_gap"] = {
                                "start_s": previous_time,
                                "end_s": packet.timestamp_seconds,
                                "reason": "source_or_sampling_or_pending_replacement_gap",
                            }
                        if sequence == 0:
                            first_publication.update(
                                warm_ms=(published - origin) * 1000,
                                cold_ms=(published - started) * 1000,
                            )
                        # Both callbacks receive exactly the same source/sequence.
                        # JPEG is ready before publication, no stale-frame overlay.
                        if on_frame is not None:
                            on_frame(jpeg, row)
                        on_update(row)
                        timing["publication_callbacks_ms"] = max(0, clock() - published) * 1000
                        _line(output, row)
                        _line(latencies, timing)
                        counts["frames_published"] += 1
                        counts["observations_without_detections"] += not bool(result["detections"])
                        previous_time, previous_index = packet.timestamp_seconds, packet.frame_index
                        sequence += 1
                        # Event consumers do not inflate frame callback timing.
                        deliver_events(row["action_events"])
                        if result.get("runtime"):
                            manifest["runtime"] = result["runtime"]
            except (InterruptedError, MediaCancelled):
                stop.set()
            except Exception as exc:
                errors.append(exc)
                stop.set()

        worker = threading.Thread(target=inference, name="recorded-replay-inference")
        worker.start()
        sampler = TimeSampler(config.sample_fps)
        previous_pts_time = -1.0
        decoded = frames(source, metrics, stop, cancel=stop.is_set, decoder=config.decoder)
        try:
            for packet in decoded:
                _check_stop(stop)
                verify_inputs()
                if packet.timestamp_seconds <= previous_pts_time or packet.source_pts is None:
                    raise ValueError("Replay yêu cầu PTS nguồn tăng nghiêm ngặt.")
                previous_pts_time = packet.timestamp_seconds
                due = origin + packet.timestamp_seconds
                while clock() < due and not stop.is_set():
                    wait(stop, min(0.05, due - clock()))
                _check_stop(stop)
                counts["frames_released"] += 1
                if sampler.select(packet.timestamp_seconds):
                    counts["frames_selected"] += 1
                    mailbox.offer((packet, due, clock()))
        finally:
            decoded.close()
            mailbox.finish()
        while worker.is_alive():
            worker.join(0.05)
        verify_inputs()
        status = "failed" if errors else "cancelled" if stop.is_set() else "completed"
    except (InterruptedError, MediaCancelled):
        status = "cancelled"
    except Exception as exc:
        errors.append(exc)
        status = "failed"
    finally:
        if worker is not None and worker.is_alive():
            stop.set()
            mailbox.finish()
            worker.join()
        if errors:
            status = "failed"
        if action_state is not None:
            try:
                events = action_state.finish(
                    reason="source_end" if status == "completed" else status
                )
                if events:
                    deliver_events(record_events(events, last_action_observation, terminal=True))
            except Exception as exc:
                errors.append(exc)
                status = "failed"
        action_summary["observation_counts"] = (
            action_state.observation_counts if action_state is not None else {}
        )
        if tracker is not None:
            tracker.finish()
        counts["pending_replaced"] = mailbox.replaced
        counts["frames_not_selected"] = counts["frames_released"] - counts["frames_selected"]
        counts["selected_not_published"] = counts["frames_selected"] - counts["frames_published"]
        manifest["status"] = status
        _json(directory / "manifest.json", manifest)
        report = {
            "schema_version": 2,
            "mode": "recorded_stream_replay_1x",
            "status": status,
            "error": str(errors[0]) if errors else None,
            "action_status": action_status,
            "action_rules": action_rules,
            "action_summary": action_summary,
            "requested_device": config.device,
            "decoder": manifest.get("decoder"),
            "resolved_device": manifest["resolved_device"],
            "requested_sample_fps": config.sample_fps,
            "effective_nominal_sample_fps": effective_fps,
            "counts": counts,
            "preparation": preparation,
            "source_clock_origin_monotonic": origin,
            "total_cold_elapsed_seconds": clock() - started,
            "source_clock_elapsed_seconds": clock() - origin if origin is not None else None,
            "pending_capacity": 1,
            "first_update_warm_ms": first_publication["warm_ms"],
            "first_update_cold_ms": first_publication["cold_ms"],
            "web_presentation_latency_ms": None,
            "latency": _lifetime_latency(latency_path),
            "tracking_metrics": metrics.report(),
            "notes": "Recorded local source replayed causally at 1x PTS, not camera/live. Cold preflight/hash/export/compile reported separately. Optional raise_hand/basic_pose are uncalibrated rule/draft proxies, not GT or all-four behavior recognition. Board arrival requires a source-bound human-selected floor ROI. Observed event boundaries/history differ from availability timing. JPEG/results share sequence and source frame. Event callback cost is separate from frame publication; no browser presentation measurement. Native calls check cancellation when they return.",
        }
        _json(directory / "report.json", report)
    return directory
