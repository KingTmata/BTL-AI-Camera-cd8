"""Run from repo root: python scripts/data/week2.py --help."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset import ROOT, assign_splits, inventory, lock_release, read_manifest, validate, verify_release, write_manifest


def main():
    parser = argparse.ArgumentParser(description="Validate, group-split and lock project8 dataset")
    parser.add_argument("command", choices=("inventory", "check", "split", "lock", "verify"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--release", type=Path, default=ROOT / "data/dataset/v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--source-id")
    parser.add_argument("--session-id", default="unknown")
    args = parser.parse_args()
    try:
        if args.command == "inventory":
            if not args.images or not args.labels or not args.source_id or not args.output or args.output.exists():
                raise ValueError("inventory requires --images, --labels, --source-id and NEW --output")
            write_manifest(args.output, inventory(args.images, args.labels, args.source_id, args.session_id))
            print(f"Pending manifest created: {args.output}; complete metadata and review before lock.")
            return
        if args.command == "verify":
            print(json.dumps({"valid": True, "release_sha256": verify_release(args.release)}))
            return
        rows = read_manifest(args.manifest)
        report, enriched = validate(rows)
        if not report["valid"]:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            raise SystemExit(1)
        if args.command == "split":
            if args.output is None or args.output.exists():
                raise ValueError("--output must be a new CSV path")
            write_manifest(args.output, assign_splits(enriched, args.val_fraction, args.seed))
        elif args.command == "lock":
            report = lock_release(rows, args.release, seed=args.seed)
        elif args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
