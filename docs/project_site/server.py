"""Small local server for the project overview site and video metadata probe."""

from __future__ import annotations

import argparse
import ast
import ipaddress
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
SITE_ROOT = Path(__file__).resolve().parent
STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/style.css": "style.css",
    "/assets/overview.svg": "assets/overview.svg",
    "/assets/project-overview.png": "assets/project-overview.png",
}
DOCUMENTS = {
    "memory": ("tasks/project_memory.md", "Ngữ cảnh và quyết định dự án", 6500),
    "video": ("docs/guides/video_processing.md", "Xử lý video local", 24000),
    "annotation": ("tasks/datasets/classroom_dataset_annotation_plan_20261015.md", "Gán nhãn dữ liệu lớp học", 24000),
    "tracking": ("tasks/video/video_tracking_plan_20261015.md", "Theo dõi người trong video", 24000),
    "architecture": ("docs/architecture/project_structure.md", "Cấu trúc dự án", 24000),
}
MAX_BODY_BYTES = 16 * 1024
MAX_CODE_BYTES = 80 * 1024

# A deliberately small, reviewed map of public application and project-site
# source. Never turn an API value into a filesystem path.
CODE_FILES = (
    ("video-cli", "Lệnh video", "src/classroom_ai/video/__main__.py", "Video", "Nhận lệnh CLI/TUI, chọn chế độ xử lý và gọi pipeline video.", "Tham số dòng lệnh, video, cấu hình detector", "Tiến trình xử lý và báo cáo kết quả"),
    ("video-core", "Nền tảng video", "src/classroom_ai/video/core.py", "Video", "Chứa cấu hình, trạng thái frame và các tiện ích lõi dùng chung.", "Cấu hình video và frame", "Trạng thái, bộ đệm và dữ liệu dùng chung"),
    ("video-media", "Đọc và ghi video", "src/classroom_ai/video/media.py", "Video", "Mở nguồn video, kiểm tra metadata và quản lý đầu ra media.", "Đường dẫn video và thông tin luồng", "Frame cùng metadata đầu ra"),
    ("video-pipeline", "Pipeline xử lý", "src/classroom_ai/video/pipeline.py", "Video", "Điều phối đọc frame, nhận diện, ghi kết quả và thống kê.", "Video, cấu hình nhận diện", "Video kết quả và thống kê lượt chạy"),
    ("video-tracking", "Theo dõi đối tượng", "src/classroom_ai/video/tracking.py", "Video", "Quản lý ID track cục bộ cho các detection qua frame.", "Detection theo frame", "Track và ID cục bộ"),
    ("video-tracked-processing", "Xử lý có tracking", "src/classroom_ai/video/tracked_processing.py", "Video", "Kết nối tracking với luồng xử lý video ở chế độ hỗ trợ.", "Frame và detection", "Frame có thông tin track"),
    ("video-rendering", "Vẽ kết quả", "src/classroom_ai/video/rendering.py", "Video", "Vẽ detection, track và thông tin tiến trình lên frame.", "Frame, detection, track", "Frame hiển thị hoặc xuất"),
    ("video-playback", "Phát và tua video", "src/classroom_ai/video/playback.py", "Video", "Điều khiển phát, pause và tua trong luồng xem video.", "Frame đã xử lý và thao tác bàn phím", "Trạng thái phát và vị trí frame"),
    ("video-metrics", "Đo hiệu năng", "src/classroom_ai/video/metrics.py", "Video", "Tính các số đo thời gian và tài nguyên trong lượt xử lý.", "Mốc thời gian và số liệu tiến trình", "Thống kê hiệu năng"),
    ("inference-detector", "Bộ phát hiện", "src/classroom_ai/inference/detector.py", "Nhận diện", "Tải và gọi detector để tạo detection từ ảnh/frame.", "Ảnh/frame, weights và tham số", "Detection và thông tin mô hình"),
    ("terminal-ui", "Giao diện terminal", "src/classroom_ai/interfaces/terminal.py", "Giao diện", "Cung cấp giao diện terminal để nhập tùy chọn xử lý video.", "Thao tác người dùng", "Cấu hình và yêu cầu chạy"),
    ("terminal-process", "Tiến trình terminal", "src/classroom_ai/interfaces/terminal_process.py", "Giao diện", "Quản lý tiến trình backend và chuyển log cho giao diện terminal.", "Cấu hình chạy và yêu cầu dừng", "Log và trạng thái tiến trình"),
    ("web-app", "Web video và timeline", "src/classroom_ai/interfaces/web.py", "Giao diện", "Điểm vào web local dùng chung pipeline Detect/Pose/tracking và duyệt sự kiện.", "Video local, cấu hình và thao tác duyệt", "Lượt xử lý, video kết quả và tổng hợp sự kiện"),
    ("streamlit-tools", "Công cụ Streamlit nghiên cứu", "src/classroom_ai/interfaces/streamlit_ui/app.py", "Giao diện", "Giữ công cụ ảnh/webcam/Training của giai đoạn trước.", "Ảnh/webcam và cấu hình nghiên cứu", "Kết quả nhận diện và huấn luyện"),
    ("training-manager", "Điều phối huấn luyện", "src/classroom_ai/training/manager.py", "Huấn luyện", "Quản lý cấu hình và thao tác huấn luyện của ứng dụng.", "Dataset, cấu hình và thiết bị", "Trạng thái cùng kết quả huấn luyện"),
    ("site-page", "Trang thuyết minh", "docs/project_site/index.html", "Website dự án", "Khung trang web thuyết minh chạy trên máy local.", "Dữ liệu trang và tương tác", "Nội dung hiển thị trong trình duyệt"),
    ("site-interactions", "Sơ đồ và tương tác website", "docs/project_site/app.js", "Website dự án", "Khai báo sơ đồ, nội dung tương tác và kết nối dữ liệu thuyết minh.", "Dữ liệu dự án và thao tác người xem", "Sơ đồ và các phần nội dung tương tác"),
    ("site-styles", "Giao diện website", "docs/project_site/style.css", "Website dự án", "Định dạng bố cục và cách hiển thị trang thuyết minh.", "Các thành phần HTML", "Giao diện trình bày"),
    ("site-server", "Máy chủ website local", "docs/project_site/server.py", "Website dự án", "Phục vụ trang thuyết minh và các API local được giới hạn.", "Yêu cầu HTTP local", "Trang, metadata và nội dung nguồn được cho phép"),
    ("site-launcher", "Mở website local", "run_project_site.cmd", "Website dự án", "Launcher Windows để chạy website thuyết minh local.", "Lệnh chạy launcher", "Máy chủ website local"),
)
CODE_BY_ID = {item[0]: item for item in CODE_FILES}


def is_loopback_address(address: str) -> bool:
    try:
        return ipaddress.ip_address(address.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def resolve_static_path(url_path: str) -> Path | None:
    path = unquote(url_path.split("?", 1)[0])
    if "\\" in path or "\x00" in path or any(part == ".." for part in path.split("/")):
        return None
    relative = STATIC_FILES.get(path)
    if relative is None:
        return None
    candidate = (SITE_ROOT / relative).resolve()
    try:
        candidate.relative_to(SITE_ROOT.resolve())
    except ValueError:
        return None
    return candidate


def probe_access_error(client_ip: str, origin: str | None, host: str, content_type: str):
    if not is_loopback_address(client_ip):
        return 403, {"ok": False, "code": "loopback_only", "error": "Đọc tệp local chỉ được phép từ chính máy chủ."}
    parsed_origin = urlsplit(origin or "")
    origin_host = (parsed_origin.hostname or "").lower()
    try:
        origin_is_loopback = ipaddress.ip_address(origin_host).is_loopback
    except ValueError:
        origin_is_loopback = origin_host == "localhost"
    if (
        parsed_origin.scheme != "http"
        or not origin_is_loopback
        or parsed_origin.netloc.lower() != host.lower()
        or parsed_origin.path not in ("", "/")
        or parsed_origin.query
        or parsed_origin.fragment
    ):
        return 403, {"ok": False, "code": "origin_rejected", "error": "Yêu cầu phải đến từ trang local cùng địa chỉ."}
    if content_type != "application/json":
        return 415, {"ok": False, "code": "json_required", "error": "Gửi dữ liệu dạng application/json."}
    return None


def extract_official_project(markdown: str) -> dict:
    """Read the one official project section, stopping before historical plans."""
    start = re.search(r"(?m)^## Mục tiêu và yêu cầu chính thức — 15/10/2026\s*$", markdown)
    if not start:
        raise ValueError("Không tìm thấy mục tiêu chính thức trong tasks/plan.md.")
    tail = markdown[start.end():]
    next_section = re.search(r"(?m)^## ", tail)
    section = tail[: next_section.start()] if next_section else tail
    title_match = re.search(r"(?m)^### Tên đề tài\s*\n+(.+?)\s*(?:\n|$)", section)
    objective_match = re.search(
        r"(?ms)^### Mục tiêu đề tài\s*\n+(.*?)(?=^### |\Z)", section
    )
    items_match = re.search(
        r"(?ms)^### Nội dung nghiên cứu\s*\n+(.*?)(?=^### |\Z)", section
    )
    if not (title_match and objective_match and items_match):
        raise ValueError("Mục tiêu chính thức thiếu tên đề tài, mục tiêu hoặc nội dung nghiên cứu.")
    research_items = re.findall(r"(?m)^\d+\.\s+(.+)$", items_match.group(1))
    return {
        "title": title_match.group(1).strip(),
        "objective": " ".join(objective_match.group(1).split()),
        "research_items": research_items,
        "section": section.strip(),
    }


def normalize_probe_result(stdout: str, filename: str, file_size: int) -> dict:
    payload = json.loads(stdout)
    fmt = payload.get("format") or {}
    streams = payload.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("Tệp không có luồng video.")
    names = str(fmt.get("format_name") or "unknown")
    container = names.split(",", 1)[0]
    codec = video.get("codec_name")
    level = _optional_int(video.get("level"))
    level_divisor = 30 if codec == "hevc" else (10 if codec in {"h264", "avc1"} else None)
    level_label = f"L{level / level_divisor:.1f}" if level is not None and level_divisor else None
    try:
        fps_numerator, fps_denominator = str(video.get("avg_frame_rate", "0/1")).split("/", 1)
        fps = float(fps_numerator) / float(fps_denominator)
    except (ValueError, ZeroDivisionError):
        fps = None
    warnings = []
    extension = Path(filename).suffix.lower()
    if extension == ".mp4" and not any(name in names.lower() for name in ("mp4", "mov")):
        warnings.append(f"Container {container} differs from filename extension .mp4.")
    duration = _optional_float(fmt.get("duration") or video.get("duration"))
    fmt_size = _optional_int(fmt.get("size"))
    return {
        "ok": True,
        "name": Path(filename).name,
        "container": container,
        "container_formats": names,
        "major_brand": (fmt.get("tags") or {}).get("major_brand"),
        "container_long_name": fmt.get("format_long_name"),
        "codec": codec,
        "profile": video.get("profile"),
        "level": level,
        "level_label": level_label,
        "width": _optional_int(video.get("width")),
        "height": _optional_int(video.get("height")),
        "fps": fps,
        "duration_seconds": duration,
        "size_bytes": fmt_size or file_size,
        "bit_rate": _optional_int(fmt.get("bit_rate")),
        "video_bit_rate": _optional_int(video.get("bit_rate")),
        "pixel_format": video.get("pix_fmt"),
        "color_range": video.get("color_range"),
        "color_space": video.get("color_space"),
        "streams": [
            {
                "type": stream.get("codec_type"),
                "codec": stream.get("codec_name"),
                "profile": stream.get("profile"),
                "width": _optional_int(stream.get("width")),
                "height": _optional_int(stream.get("height")),
                "channels": _optional_int(stream.get("channels")),
                "language": (stream.get("tags") or {}).get("language"),
            }
            for stream in streams
        ],
        "warnings": warnings,
    }


def _optional_float(value):
    try:
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _optional_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _find_ffprobe() -> str | None:
    found = shutil.which("ffprobe")
    if found:
        return found
    try:
        import imageio_ffmpeg

        sibling = Path(imageio_ffmpeg.get_ffmpeg_exe()).with_name(
            "ffprobe.exe" if os.name == "nt" else "ffprobe"
        )
        if sibling.is_file():
            return str(sibling)
    except (ImportError, OSError, RuntimeError):
        pass
    return None


def probe_local_video(path: str, ffprobe: str | None = None) -> dict:
    path = path.strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in {"'", '"'}:
        path = path[1:-1].strip()
    if "\x00" in path:
        return {"ok": False, "code": "invalid_path", "error": "Đường dẫn có ký tự không hợp lệ."}
    if path.startswith(("\\\\", "//")):
        return {"ok": False, "code": "unsupported_path", "error": "Đường dẫn UNC/mạng không được hỗ trợ; chọn tệp local trên máy chủ."}
    try:
        candidate = Path(path).expanduser().resolve()
        info = candidate.stat()
    except FileNotFoundError:
        return {"ok": False, "code": "file_not_found", "error": f"Không tìm thấy tệp: {path}"}
    except PermissionError:
        return {"ok": False, "code": "file_access_denied", "error": "Không có quyền đọc đường dẫn này trên máy chủ."}
    except (OSError, RuntimeError, ValueError) as exc:
        return {"ok": False, "code": "invalid_path", "error": f"Không thể đọc đường dẫn tệp: {exc}"}
    if not stat.S_ISREG(info.st_mode):
        return {"ok": False, "code": "not_a_file", "error": "Đường dẫn cần trỏ trực tiếp đến một tệp video."}
    executable = ffprobe if ffprobe is not None else _find_ffprobe()
    if not executable:
        return {
            "ok": False,
            "code": "ffprobe_unavailable",
            "error": "Không tìm thấy ffprobe. Cài FFmpeg và thêm ffprobe vào PATH để đọc metadata video.",
        }
    try:
        result = subprocess.run(
            [executable, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(candidate)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": "probe_timeout", "error": "Đọc metadata quá 30 giây và đã dừng."}
    except OSError as exc:
        return {"ok": False, "code": "probe_failed", "error": f"Không khởi chạy được ffprobe: {exc}"}
    if result.returncode:
        detail = result.stderr.strip()[-1000:]
        return {"ok": False, "code": "probe_failed", "error": detail or "ffprobe không đọc được tệp này."}
    try:
        metadata = normalize_probe_result(result.stdout, candidate.name, candidate.stat().st_size)
        metadata["path"] = str(candidate)
        return metadata
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        return {"ok": False, "code": "invalid_media", "error": str(exc) or "ffprobe trả metadata không hợp lệ."}


def _sanitize_document(text: str, limit: int) -> tuple[str, bool]:
    text = re.sub(r"(?i)[A-Z]:\\(?:[^\s`|<>]+\\?)*", "[đường dẫn local]", text)
    truncated = len(text) > limit
    if truncated:
        marker = "\n\n[Đã rút gọn; mở tài liệu nguồn để đọc đầy đủ.]"
        text = text[: max(0, limit - len(marker))].rstrip() + marker
    return text.strip(), truncated


def _safe_code_path(relative: str) -> Path | None:
    """Resolve only curated source paths and reject symlinks at every level."""
    candidate = ROOT
    for part in Path(relative).parts:
        candidate = candidate / part
        try:
            if candidate.is_symlink():
                return None
        except OSError:
            return None
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(ROOT.resolve())
        if not resolved.is_file():
            return None
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None


def _code_symbols(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    return [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ][:40]


def code_catalog() -> dict:
    modules = []
    for item in CODE_FILES:
        module_id, title, relative, group, purpose, inputs, outputs = item
        path = _safe_code_path(relative)
        exists = path is not None
        symbols: list[str] = []
        line_count = 0
        if path is not None:
            try:
                source = path.read_text(encoding="utf-8")
                line_count = len(source.splitlines())
                symbols = _code_symbols(source)
            except (OSError, UnicodeError):
                exists = False
                line_count = 0
        modules.append({
            "id": module_id, "title": title, "path": relative,
            "group": {"Video": "video", "Nhận diện": "inference", "Giao diện": "interface", "Huấn luyện": "training", "Website dự án": "site"}.get(group, group),
            "purpose": purpose, "inputs": inputs, "outputs": outputs,
            "symbols": symbols, "file_exists": exists, "line_count": line_count,
        })
    return {"modules": modules}


def code_source(module_id: str) -> tuple[int, dict]:
    item = CODE_BY_ID.get(module_id)
    if item is None:
        return 404, {"ok": False, "code": "not_found", "error": "Không tìm thấy mã nguồn được phép xem."}
    path = _safe_code_path(item[2])
    if path is None:
        return 404, {"ok": False, "code": "not_found", "error": "Không tìm thấy mã nguồn được phép xem."}
    try:
        raw = path.read_bytes()
        clipped = len(raw) > MAX_CODE_BYTES
        content = raw[:MAX_CODE_BYTES].decode("utf-8", errors="replace")
        return 200, {
            "id": module_id, "path": item[2], "content": content,
            "line_count": len(content.splitlines()), "truncated": clipped,
        }
    except OSError:
        return 404, {"ok": False, "code": "not_found", "error": "Không tìm thấy mã nguồn được phép xem."}


def project_payload() -> dict:
    official = extract_official_project((ROOT / "tasks" / "plan.md").read_text(encoding="utf-8"))
    official_content, official_truncated = _sanitize_document(official["section"], 9000)
    sources = [{
        "id": "plan",
        "title": "Mục tiêu chính thức",
        "path": "tasks/plan.md#mục-tiêu-và-yêu-cầu-chính-thức--15102026",
        "content": official_content,
        "truncated": official_truncated,
    }]
    for source_id, (relative, title, limit) in DOCUMENTS.items():
        try:
            content = (ROOT / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if source_id == "memory":
            match = re.search(r"(?ms)^## Những điều đã thống nhất\s*\n+(.*?)(?=^## |\Z)", content)
            content = match.group(0) if match else content
            current_scope = re.search(r"(?m)^- 22/10/2026: (.+)$", (ROOT / relative).read_text(encoding="utf-8"))
            if current_scope:
                content += "\n\n### Phạm vi cập nhật 22/10/2026\n- " + current_scope.group(1)
            pilot_note = (
                "### Thử nghiệm gán nhãn ngoài ứng dụng\n"
                "- Đã thử Label Studio 1.23.2 với backend YOLO26l/YOLO26x cho lớp person. "
                "Bản snapshot có 2 project, 3 task, 1 annotation thủ công và 3 prediction AI; "
                "prediction chưa được duyệt nên không được xem là ground truth. Đây là pilot hỗ trợ "
                "gán nhãn riêng, không phải bộ phận của ứng dụng camera và không chốt model nghiên cứu. "
                "Không chạy thêm model sau yêu cầu hạn chế test; cần export hoặc sao lưu mới khi bổ sung nhãn."
            )
            content = pilot_note + "\n\n" + content
        safe_content, truncated = _sanitize_document(content, limit)
        sources.append({
            "id": source_id,
            "title": title,
            "path": relative.replace("\\", "/"),
            "content": safe_content,
            "truncated": truncated,
        })
    return {
        "title": official["title"],
        "objective": official["objective"],
        "research_items": official["research_items"],
        "sources": sources,
    }


class ProjectSiteHandler(SimpleHTTPRequestHandler):
    server_version = "ProjectSite/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE_ROOT), **kwargs)

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {self.client_address[0]} {fmt % args}")

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path == "/api/health":
            ffprobe = _find_ffprobe()
            return self._json(200, {
                "ok": True,
                "probe_available": bool(ffprobe),
                "probe_backend": "ffprobe" if ffprobe else None,
                "host": self.server.bind_host,
                "port": self.server.server_port,
            })
        if parsed.path == "/api/project":
            try:
                return self._json(200, project_payload())
            except (OSError, UnicodeError, ValueError) as exc:
                return self._json(500, {"ok": False, "code": "project_content_error", "error": str(exc)})
        if parsed.path == "/api/code":
            return self._json(200, code_catalog())
        if parsed.path.startswith("/api/code/"):
            module_id = parsed.path[len("/api/code/"):]
            if not module_id or "/" in module_id or "\\" in module_id:
                return self._json(404, {"ok": False, "code": "not_found", "error": "Không tìm thấy mã nguồn được phép xem."})
            status, payload = code_source(module_id)
            return self._json(status, payload)
        local_path = resolve_static_path(parsed.path)
        if local_path is None or not local_path.is_file():
            return self._json(404, {"ok": False, "code": "not_found", "error": "Không tìm thấy tài nguyên."})
        if local_path.suffix.lower() not in {".html", ".js", ".css", ".svg", ".png"}:
            return self._json(404, {"ok": False, "code": "not_found", "error": "Không tìm thấy tài nguyên."})
        body = local_path.read_bytes()
        content_type = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png"}[local_path.suffix.lower()]
        self.send_response(200)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        local_path = resolve_static_path(urlsplit(self.path).path)
        if local_path is None or not local_path.is_file() or local_path.suffix.lower() not in {".html", ".js", ".css", ".svg", ".png"}:
            self.send_error(404, "Not found")
            return
        content_type = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png"}[local_path.suffix.lower()]
        self.send_response(200)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(local_path.stat().st_size))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

    def do_POST(self):
        self.close_connection = True
        parsed = urlsplit(self.path)
        if parsed.path != "/api/probe":
            return self._json(404, {"ok": False, "code": "not_found", "error": "Endpoint không tồn tại."})
        origin = self.headers.get("Origin")
        access_error = probe_access_error(
            self.client_address[0], origin, self.headers.get("Host", ""), self.headers.get_content_type()
        )
        if access_error:
            return self._json(*access_error)
        try:
            size = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            size = -1
        if size < 0 or size > MAX_BODY_BYTES:
            return self._json(413, {"ok": False, "code": "payload_size", "error": "Dữ liệu yêu cầu tối đa 16 KiB."})
        try:
            request = json.loads(self.rfile.read(size))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._json(400, {"ok": False, "code": "invalid_json", "error": "JSON không hợp lệ."})
        path = request.get("path") if isinstance(request, dict) else None
        if not isinstance(path, str) or not path.strip() or len(path) > 4096:
            return self._json(400, {"ok": False, "code": "invalid_path", "error": "Nhập đường dẫn tệp video trên máy đang chạy website."})
        result = probe_local_video(path.strip())
        return self._json(200 if result.get("ok") else 422, result)

    def translate_path(self, path):
        # Keep SimpleHTTPRequestHandler from exposing its default directory listing.
        return super().translate_path(path)


class ProjectSiteServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, bind_host):
        self.bind_host = bind_host
        super().__init__(address, handler)


def _lan_addresses() -> list[str]:
    addresses = set()
    try:
        for item in socket.getaddrinfo(
            socket.gethostname(), None, family=socket.AF_INET, type=socket.SOCK_STREAM
        ):
            address = item[4][0]
            if _is_usable_lan_ipv4(address):
                addresses.add(address)
    except OSError:
        pass
    return sorted(addresses)


def _is_usable_lan_ipv4(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
        return (
            isinstance(parsed, ipaddress.IPv4Address)
            and not parsed.is_loopback
            and not parsed.is_link_local
            and not parsed.is_unspecified
            and not parsed.is_multicast
        )
    except ValueError:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the local project overview and video metadata probe.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.port < 65536:
        parser.error("--port phải nằm trong khoảng 1–65535.")
    server = ProjectSiteServer((args.host, args.port), ProjectSiteHandler, args.host)
    port = server.server_port
    print(f"Project overview: http://127.0.0.1:{port}/")
    if args.host in {"0.0.0.0", "::"}:
        for address in _lan_addresses():
            print(f"LAN: http://{address}:{port}/")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang dừng website local…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
