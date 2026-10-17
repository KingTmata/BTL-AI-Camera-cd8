"""Minimal clickable terminal launcher; frames stay in the child backend."""

import time
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DirectoryTree,
    Footer,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
)

from classroom_ai.interfaces.terminal_process import ROOT, BackendJob, progress_fraction


class VideoTree(DirectoryTree):
    def filter_paths(self, paths):
        extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}
        return [path for path in paths if path.is_dir() or path.suffix.lower() in extensions]


class VideoPicker(ModalScreen):
    BINDINGS = [("escape", "cancel", "Đóng")]
    CSS = """
    VideoPicker { align: center middle; }
    #picker { width: 90%; height: 85%; border: solid $accent; padding: 1; background: $surface; }
    #folder-row { height: 3; }
    #folder { width: 1fr; }
    #folder-open { width: 14; min-width: 10; }
    DirectoryTree { height: 1fr; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="picker"):
            yield Label("Chọn video — mở thư mục, rồi bấm vào file")
            with Horizontal(id="folder-row"):
                yield Input(str(ROOT), id="folder")
                yield Button("Mở thư mục", id="folder-open")
            yield VideoTree(ROOT, id="files")
            yield Button("Đóng", id="cancel")

    def action_cancel(self):
        self.dismiss(None)

    def on_mount(self):
        self.query_one(Input).cursor_blink = False

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "folder-open":
            self.open_folder()

    def on_input_submitted(self, event: Input.Submitted):
        self.open_folder()

    def open_folder(self):
        path = Path(self.query_one("#folder", Input).value.strip().strip('"')).expanduser()
        if path.is_dir():
            self.query_one(DirectoryTree).path = path
        else:
            self.notify("Thư mục không tồn tại.", severity="error")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected):
        event.stop()
        self.dismiss(event.path)


class VideoTerminal(App):
    TITLE = "AI Camera · Video"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        ("ctrl+r", "start", "Chạy"),
        ("ctrl+d", "stop", "Dừng"),
        ("ctrl+q", "safe_quit", "Thoát"),
        ("ctrl+c", "safe_quit", "Thoát"),
    ]
    CSS = """
    Screen { background: $surface; }
    #main { padding: 1 2; height: 1fr; }
    #title { height: 2; text-style: bold; }
    #source-row { height: 3; }
    #source { width: 1fr; }
    #browse { width: 12; min-width: 10; margin-left: 1; }
    #actions { height: 3; margin-top: 1; }
    #mode { width: 1fr; margin-right: 1; }
    #start, #stop { width: 12; min-width: 10; margin-left: 1; }
    #hint { height: 2; color: $text-muted; }
    #status { height: auto; min-height: 2; margin-top: 1; }
    ProgressBar { height: 1; margin-bottom: 1; }
    RichLog { border: solid $primary-muted; height: 1fr; min-height: 4; }
    """

    def __init__(self, source="", mode="fast", options=None, job=None):
        super().__init__()
        self.animation_level = "none"
        self.initial_source, self.initial_mode = str(source), mode
        self.options = options or {}
        self.job = job if job is not None else BackendJob()
        self.was_running = False
        self.exit_when_done = False
        self.exit_deadline = None

    def compose(self) -> ComposeResult:
        with Vertical(id="main"):
            yield Static("AI CAMERA / VIDEO", id="title")
            with Horizontal(id="source-row"):
                yield Input(self.initial_source, placeholder="Đường dẫn video…", id="source")
                yield Button("Chọn…", id="browse")
            with Horizontal(id="actions"):
                yield Select(
                    [("Nhanh · lấy mẫu", "fast"), ("Đầy đủ · xuất rồi phát", "full")],
                    value=self.initial_mode,
                    allow_blank=False,
                    id="mode",
                )
                yield Button("Chạy", variant="primary", id="start")
                yield Button("Dừng", id="stop", disabled=True)
            yield Static(
                "OpenCV: Space tạm dừng. Full: kéo thanh tua hoặc A/D ±5 giây; Q để đóng.",
                id="hint",
            )
            yield Checkbox(
                "Bám người (Full) · ID/màu cố định, nội suy tối đa 0,3s",
                value=self.initial_mode == "full" and self.options.get("tracking") == "bytetrack",
                disabled=self.initial_mode != "full",
                id="tracking",
            )
            yield Static("Sẵn sàng. Chọn video rồi bấm Chạy.", id="status", markup=False)
            yield ProgressBar(total=100, show_eta=False, show_percentage=False)
            yield RichLog(
                id="log", max_lines=200, min_width=20, wrap=True, markup=False, highlight=False
            )
        yield Footer()

    def on_mount(self):
        self.query_one(Input).cursor_blink = False
        self.set_interval(1, self.poll_job)

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "browse":
            self.push_screen(VideoPicker(), self.selected_file)
        elif event.button.id == "start":
            self.action_start()
        elif event.button.id == "stop":
            self.action_stop()

    def selected_file(self, path):
        if path is not None:
            self.query_one("#source", Input).value = str(path)

    def action_start(self):
        if self.job.running:
            return
        raw = self.query_one("#source", Input).value.strip().strip('"')
        if not raw or not Path(raw).expanduser().is_file():
            self.query_one("#status", Static).update("Chọn một file video có thật trước khi chạy.")
            return
        try:
            mode = self.query_one("#mode", Select).value
            options = dict(self.options)
            if mode == "full":
                options["tracking"] = (
                    "bytetrack" if self.query_one("#tracking", Checkbox).value else "off"
                )
            else:
                options = {
                    k: v
                    for k, v in options.items()
                    if k != "tracking" and not k.startswith("track_")
                }
            self.job.start(Path(raw), mode, options)
        except (ValueError, OSError) as exc:
            self.query_one("#status", Static).update(str(exc))
            return
        self.was_running = True
        self.query_one("#log", RichLog).clear()
        self.query_one("#status", Static).update("Đang khởi động backend…")
        # No indeterminate animation: the loading stage is reported as text.
        self.query_one(ProgressBar).update(total=100, progress=0)
        self.set_running(True)

    def set_running(self, running):
        for selector in ("#start", "#browse", "#source", "#mode"):
            self.query_one(selector).disabled = running
        self.query_one("#stop", Button).disabled = not running
        self.query_one("#tracking", Checkbox).disabled = (
            running or self.query_one("#mode", Select).value != "full"
        )

    def on_select_changed(self, event: Select.Changed):
        if event.select.id == "mode":
            checkbox = self.query_one("#tracking", Checkbox)
            checkbox.disabled = self.job.running or event.value != "full"
            if event.value != "full":
                checkbox.value = False

    def action_stop(self):
        if self.job.running:
            self.job.request_stop()
            self.query_one("#status", Static).update(
                "Đang dừng backend, đợi tác vụ hiện tại kết thúc…"
            )
            self.query_one("#stop", Button).disabled = True

    def poll_job(self):
        # Textual stops the app before removing its screen/widgets. A queued
        # timer must not query the DOM while that teardown is in progress.
        if not self.is_running:
            return
        for line in self.job.read_updates():
            self.query_one("#log", RichLog).write(line)
            fraction = progress_fraction(line)
            if fraction:
                self.query_one(ProgressBar).update(total=fraction[1], progress=fraction[0])
            if " | " in line and not line.startswith("  "):
                self.query_one("#status", Static).update(line)
        if self.was_running and not self.job.running:
            self.was_running = False
            self.set_running(False)
            code = self.job.returncode
            status = (
                "Hoàn tất"
                if code == 0
                else "Đã dừng"
                if code == 130
                else f"Backend báo lỗi ({code})"
            )
            self.query_one("#status", Static).update(f"{status}. Kết quả: {self.job.result_dir}")
            if code == 0:
                self.query_one(ProgressBar).update(total=100, progress=100)
            else:
                self.query_one(ProgressBar).update(total=100, progress=0)
            self.job.close()
        if self.exit_when_done:
            if self.job.running and time.monotonic() >= self.exit_deadline:
                self.job.force_stop()
            if not self.job.running:
                self.exit()

    def action_safe_quit(self):
        if not self.job.running:
            self.exit()
            return
        self.exit_when_done = True
        self.exit_deadline = time.monotonic() + 10
        self.action_stop()

    def on_unmount(self):
        self.job.close()


def main(source="", mode="fast", options=None):
    VideoTerminal(source, mode, options).run()
