"""Cài môi trường demo CPU riêng trên từng máy; --check chỉ kiểm, không cài."""

import argparse
import platform
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".venv"
PYTHON = ENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def run(*args):
    subprocess.run([str(PYTHON), *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Kiểm môi trường/weights đã có; không bật webcam")
    args = parser.parse_args()
    if not PYTHON.exists():
        if args.check:
            raise RuntimeError("Chua co .venv. Chay setup-demo.cmd (Windows) hoac python3.11 scripts/setup_demo.py.")
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError("Can Python 3.11. Tren Windows: py -3.11 scripts/setup_demo.py.")
        print("Creating local Python 3.11 environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(ENV)
    run("-c", "import sys; assert sys.version_info[:2] == (3, 11), 'Existing .venv must use Python 3.11; keep it and create the demo in a separate project folder.'")
    if not args.check:
        run("-m", "pip", "install", "--upgrade", "pip")
        torch_args = ["-m", "pip", "install", "torch==2.14.0", "torchvision==0.29.0"]
        if platform.system() in {"Windows", "Linux"}:
            torch_args += ["--index-url", "https://download.pytorch.org/whl/cpu"]
        run(*torch_args)
        # Lock đã kiểm trên Windows; macOS/Linux khóa các gói chính, không mang lock hệ điều hành khác.
        requirements = "requirements-demo-cpu.lock.txt" if sys.platform == "win32" else "requirements-demo.in"
        run("-m", "pip", "install", "-r", requirements)
        (ROOT / "weights").mkdir(exist_ok=True)
        run("-c", "from pathlib import Path; from ultralytics import YOLO; YOLO(str(Path('weights/yolo26n.pt').resolve()))")
    run("-m", "pip", "check")
    run("-c", "import cv2, streamlit, torch, ultralytics; from pathlib import Path; from src.inference.detector import ImageInspector; ImageInspector(Path('weights/yolo26n.pt')); assert hasattr(cv2, 'imshow'), 'Need opencv-python, not headless'; print('Demo imports and checkpoint OK'); print('Versions:', ultralytics.__version__, torch.__version__, streamlit.__version__, cv2.__version__)")
    print("Ready. Windows: run-demo.cmd. macOS/Linux: .venv/bin/python -m streamlit run src/app.py --server.headless=false", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        sys.exit(1)
