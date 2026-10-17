"""Synthetic detector + real full/tracker/encoder/HighGUI; not a YOLO accuracy benchmark."""

import argparse
import json
import math
from pathlib import Path

from classroom_ai.interfaces.native_video import NativeDisplay
from classroom_ai.video.pipeline import VideoConfig, run_video


class SyntheticDetector:
    def __init__(self, weights, device):
        self.calls = 0

    def predict(self, image, confidence, **kwargs):
        index = self.calls - 1  # Exclude initial warm-up.
        self.calls += 1
        x = 180 + 50 * math.sin(index / 50)
        rows = (
            []
            if index % 30 in (15, 16)
            else [
                dict(
                    class_id=0, class_name="person", confidence=0.1, bbox_xyxy=[x, 60, x + 80, 280]
                )
            ]
        )
        return dict(detections=rows, processing_ms=0)


class AutoExitDisplay(NativeDisplay):
    ended = False

    def mark_end(self):
        super().mark_end()
        self.ended = True

    def poll(self):
        return "quit" if self.ended else super().poll()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture = args.output_dir / "SYNTHETIC_DETECTOR_NOT_YOLO.fixture"
    if not fixture.exists():
        fixture.write_text(
            "Synthetic moving box with 2-frame gaps every30frames; no learned model.",
            encoding="utf-8",
        )
    config = VideoConfig(
        source=args.source,
        mode="full",
        weights=fixture,
        device="cpu",
        tracking="bytetrack",
        output_dir=args.output_dir,
    )
    directory = run_video(
        config, predictor_factory=SyntheticDetector, display_factory=AutoExitDisplay
    )
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    result = dict(
        directory=str(directory),
        source=str(args.source),
        scope="Synthetic detector, real tracker/render/encode/HighGUI. Not YOLO inference or classroom quality.",
        report=report,
    )
    (args.output_dir / "playback_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
