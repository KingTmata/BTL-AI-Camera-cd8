"""Tạo YAML/list từ cặp ảnh-nhãn COCO128; chỉ phục vụ train thử kỹ thuật."""
from pathlib import Path
import argparse

import ultralytics
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8, help="Số ảnh thử; 0 lấy mọi cặp hợp lệ")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit phải >=0")
    dataset = ROOT / "data/reference/coco128"
    images = sorted((dataset / "images/train2017").glob("*.jpg"))
    pairs = [p for p in images if (dataset / "labels/train2017" / (p.stem + ".txt")).is_file()]
    if not pairs:
        raise SystemExit("Chưa có cặp COCO128. Tải và giải nén theo README trước.")
    chosen = pairs[:args.limit] if args.limit else pairs
    output = ROOT / "runs/week1/coco128_smoke"
    output.mkdir(parents=True, exist_ok=True)
    image_list = output / "images.txt"
    image_list.write_text("".join(f"{p.as_posix()}\n" for p in chosen), encoding="utf-8")
    coco_cfg = Path(ultralytics.__file__).parent / "cfg/datasets/coco128.yaml"
    names = yaml.safe_load(coco_cfg.read_text(encoding="utf-8"))["names"]
    config = {"path": dataset.as_posix(), "train": image_list.as_posix(),
              "val": image_list.as_posix(), "names": names}
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"{len(chosen)} ảnh từ {len(pairs)} cặp. YAML: {output / 'data.yaml'}")
    print("Train và val dùng cùng ảnh: chỉ thử pipeline, không báo điểm chất lượng.")


if __name__ == "__main__":
    main()
