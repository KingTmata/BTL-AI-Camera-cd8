"""Reproduce the three full-video configurations sequentially (never uploads video)."""

import argparse
import json
from pathlib import Path

from classroom_ai.video.pipeline import VideoConfig, run_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.05)
    args = parser.parse_args()
    comparison = []
    for label, options in (
        ("detection", {}),
        ("tracking", dict(tracking="bytetrack", track_smooth_alpha=1.0, track_gap_max_ms=0)),
        ("tracking_smooth_gap", dict(tracking="bytetrack")),
    ):
        config = VideoConfig(
            source=args.source,
            weights=args.weights,
            mode="full",
            device=args.device,
            imgsz=args.imgsz,
            confidence=args.confidence,
            no_display=True,
            output_dir=args.output_dir / label,
            **options,
        )
        directory = run_video(config)
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        comparison.append(
            dict(
                configuration=label,
                directory=str(directory),
                status=report["status"],
                elapsed_seconds=report["elapsed_seconds"],
                counts=report["counts"],
                stages=report["stages"],
                rss_peak_sampled_bytes=report["rss_peak_sampled_bytes"],
                note="No ground truth or display in this comparison; box count is not recall.",
            )
        )
        # Save after every configuration, preserving completed evidence if a later run fails.
        (args.output_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
