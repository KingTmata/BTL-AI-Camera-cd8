"""Usage: python -m classroom_ai.video --source clip.mp4 --mode fast."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROGRAM_STARTED = time.perf_counter()


def main():
    parser = argparse.ArgumentParser(
        description="Local video YOLO + OpenCV. No Streamlit. Preview is silent."
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--mode", choices=("full", "fast"))
    parser.add_argument("--tui", action="store_true", help="Minimal clickable terminal interface")
    parser.add_argument("--stop-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:N")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--tracking", choices=("off", "bytetrack"), default="off")
    parser.add_argument("--track-max-lost-seconds", type=float, default=3.0)
    parser.add_argument("--track-gap-max-ms", type=float, default=300.0)
    parser.add_argument("--track-smooth-alpha", type=float, default=0.65)
    parser.add_argument("--nms-iou", type=float)
    parser.add_argument("--sample-fps", type=float, default=5)
    parser.add_argument("--buffer-ms", type=float, default=250)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--baseline", action="store_true", help="fast mode: playback without loading YOLO"
    )
    arguments = {
        key: value for key, value in vars(parser.parse_args()).items() if value is not None
    }
    if arguments.pop("tui"):
        source, mode = arguments.pop("source", ""), arguments.pop("mode", "fast")
        if arguments.get("tracking") != "off" and (mode != "full" or arguments.get("baseline")):
            parser.error("Tracking chỉ dùng với full, không dùng trong fast/baseline.")
        arguments.pop("stop_file", None)
        output_root = arguments.pop("output_dir", None)
        try:
            from classroom_ai.interfaces.terminal import VideoTerminal
            from classroom_ai.interfaces.terminal_process import BackendJob
        except ModuleNotFoundError as exc:
            if exc.name == "textual":
                print(
                    "Install terminal UI: .venv\\Scripts\\python.exe -m pip install -r requirements/video_tui.txt",
                    file=sys.stderr,
                )
                return 1
            raise
        VideoTerminal(source, mode, arguments, BackendJob(output_root)).run()
        return 0
    if "source" not in arguments or "mode" not in arguments:
        parser.error("--source và --mode là bắt buộc, hoặc dùng --tui.")
    stop_file = arguments.pop("stop_file", None)
    from classroom_ai.video.pipeline import VideoConfig, run_video

    try:
        directory = run_video(
            VideoConfig(**arguments), program_started=PROGRAM_STARTED, stop_file=stop_file
        )
        import json

        status = json.loads((directory / "report.json").read_text(encoding="utf-8"))["status"]
        return 0 if status == "completed" else 130
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
