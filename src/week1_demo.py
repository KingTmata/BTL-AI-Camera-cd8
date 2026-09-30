"""Giữ lệnh tuần 1 cũ; phần thực thi nằm tại src/inference/demo.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.demo import main

if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
