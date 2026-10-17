"""Playback controls and seeking in finalized video; inference is never repeated."""

from __future__ import annotations

import time

import cv2

from classroom_ai.video.core import Frame


class Controls:
    def __init__(self, display, stop, playing, clock, stop_file=None):
        self.display, self.stop, self.playing, self.clock = display, stop, playing, clock
        self.paused = False
        self.pause_started = None
        self.pause_seconds = 0.0
        self.stop_file = stop_file
        self.next_stop_check = 0.0

    def poll(self, allow_pause=True):
        if self.stop_file is not None:
            now = time.monotonic()
            if now >= self.next_stop_check:
                self.next_stop_check = now + 0.2
                if self.stop_file.exists():
                    self.stop.set()
        action = self.display.poll() if self.display else None
        if action == "replay" and self.paused:
            action = "pause"
        if action == "quit":
            self.stop.set()
        elif action == "pause" and allow_pause:
            self.paused = not self.paused
            now = time.perf_counter()
            if self.paused:
                self.pause_started = now
                self.playing.clear()
                self.clock.suspend(now)
            else:
                self.pause_seconds += now - self.pause_started
                self.playing.set()
                self.clock.resume(now)
        return not self.stop.is_set()

    def wait_until(self, deadline, *, interrupted=None):
        while not self.stop.is_set():
            self.poll()
            if interrupted is not None and interrupted():
                return not self.stop.is_set()
            if not self.paused and time.perf_counter() >= deadline():
                return True
            time.sleep(0.002)
        return False


def play_completed(session, source, controls, display):
    capture = cv2.VideoCapture(str(source.path), cv2.CAP_FFMPEG)
    metrics, clock = session.metrics, controls.clock
    started = time.perf_counter()
    index, seeks, shown = 0, 0, 0
    at_end = completed = False
    first_presented = last_presented = interval_start = None
    initial_pause = controls.pause_seconds
    try:
        if not capture.isOpened():
            raise ValueError("Không mở được video kết quả để phát/tua.")
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        display.enable_seek(source.frame_count, source.fps)
        session.progress.set("Playing completed video")
        clock.start(started, 0)
        while not session.stop.is_set():
            controls.poll()
            target = display.take_seek()
            preview = target is not None
            if preview:
                target = max(0, min(source.frame_count - 1, int(target)))
                before = time.perf_counter()
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, target):
                    raise RuntimeError(f"Không tua được đến frame {target}.")
                position = capture.get(cv2.CAP_PROP_POS_FRAMES)
                if abs(position - target) > 0.5:
                    raise RuntimeError(f"Decoder không định vị đúng frame {target}: {position}.")
                metrics.observe("playback_seek", (time.perf_counter() - before) * 1000)
                index, at_end = target, False
                seeks += 1
                clock.start(time.perf_counter(), target / float(source.fps))
                if controls.paused:
                    clock.suspended_at = time.perf_counter()
                interval_start = None
            if session.stop.is_set():
                break
            if (controls.paused and not preview) or at_end:
                session.progress.tick(source)
                time.sleep(0.01)
                interval_start = None
                continue
            before = time.perf_counter()
            ok, image = capture.read()
            metrics.observe("playback_decode", (time.perf_counter() - before) * 1000)
            if not ok:
                raise RuntimeError(f"Video kết quả bị ngắt tại frame {index}/{source.frame_count}.")
            metrics.count("playback_frames_decoded")
            timestamp = index / float(source.fps)
            pause_before = controls.pause_seconds
            if not preview and not controls.wait_until(
                lambda: clock.deadline(timestamp),
                interrupted=lambda: getattr(display, "seek_request", None) is not None,
            ):
                break
            requested = display.take_seek()
            if requested is not None:
                # Preserve keyboard as well as slider targets received during the wait.
                display.seek_request = requested
                continue
            if not controls.paused and time.perf_counter() - clock.deadline(timestamp) > 1 / float(
                source.fps
            ):
                metrics.count("frames_display_dropped")
            else:
                before = time.perf_counter()
                display.show(
                    Frame(index, timestamp, image),
                    None,
                    None,
                    f"full | source {float(source.fps):.3f}fps",
                )
                display.update_position(index)
                controls.poll()
                now = time.perf_counter()
                metrics.observe("draw_present", (now - before) * 1000)
                metrics.count("frames_displayed")
                if (
                    interval_start is not None
                    and pause_before == controls.pause_seconds
                    and not controls.paused
                ):
                    metrics.observe("display_interval", (now - interval_start) * 1000)
                interval_start = now
                first_presented = first_presented if first_presented is not None else now
                last_presented = now
                shown += 1
                metrics.count("frames_consumed")
            metrics.values["playback_position_frame"] = index
            index += 1
            session.progress.tick(source)
            if index >= source.frame_count:
                at_end = completed = True
                display.mark_end()
        metrics.values.update(
            playback_wall_seconds=time.perf_counter() - started,
            playback_completed=completed,
            playback_seek_count=seeks,
            user_pause_seconds=controls.pause_seconds - initial_pause,
        )
        # Seek/repeat sessions are interactive, not linear playback benchmarks.
        if seeks == 0 and shown > 1 and last_presented > first_presented:
            wall = last_presented - first_presented
            active = wall - (controls.pause_seconds - initial_pause)
            metrics.values["display_fps_wall"] = (shown - 1) / wall
            metrics.values["display_fps_excluding_user_pause"] = (
                (shown - 1) / active if active > 0 else None
            )
        elif seeks:
            metrics.values.update(
                display_fps_wall=None,
                display_fps_excluding_user_pause=None,
                playback_measurement_note="Interactive seeking/replay; FPS is not a linear-run benchmark.",
            )
    finally:
        capture.release()
