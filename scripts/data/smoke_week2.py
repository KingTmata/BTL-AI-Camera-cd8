"""Exercise Week 2 CLI and real pretrained inference using synthetic data only."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset import NAMES, ROOT, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/week2/synthetic_smoke")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        parser.error("--output must be a new directory inside repo")
    output.mkdir(parents=True)
    rows = []
    for index in range(8):
        image = output / f"synthetic_{index}.png"
        label = image.with_suffix(".txt")
        Image.new("RGB", (100, 100), (index*25, 0, 0)).save(image)
        label.write_text("\n".join(f"{i} .5 .5 .2 .2" for i in range(len(NAMES))), encoding="utf-8")
        rows.append({"image_id": f"synthetic_{index}", "image_path": image.relative_to(ROOT).as_posix(),
                     "label_path": label.relative_to(ROOT).as_posix(), "source_id": "synthetic_test_only",
                     "session_id": f"group{index//2}", "split_group_id": f"group{index//2}", "split": "",
                     "license_or_consent": "synthetic fixture", "annotator": "fixture_generator",
                     "reviewer": "fixture_assertions", "annotation_status": "approved", "review_status": "approved",
                     "near_duplicate_status": "approved", "needs_review": "no",
                     "notes": "Artificial annotations/statuses exercise software; NOT human-reviewed group data."})
    write_manifest(output / "intake.csv", rows)
    def run(*arguments):
        subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True)
    cli = "scripts/data/week2.py"
    run(cli, "check", "--manifest", output / "intake.csv")
    run(cli, "split", "--manifest", output / "intake.csv", "--output", output / "split.csv")
    run(cli, "lock", "--manifest", output / "split.csv", "--release", output / "release")
    run(cli, "verify", "--release", output / "release")
    run("-m", "src.evaluation", "--manifest", output / "release/manifest.csv", "--weights", "weights/yolo26n.pt",
        "--output", output / "evaluation")
    result = {"status": "PASS", "purpose": "synthetic software smoke; NOT dataset baseline", "classes": list(NAMES)}
    (output / "SMOKE_RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
