"""Packaging, migration integrity and UI teardown regression contracts."""

import ast
import hashlib
import json
import re
import subprocess
import sys
from unittest.mock import Mock

from classroom_ai.paths import PROJECT_ROOT


def test_installed_package_works_outside_checkout(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "from classroom_ai.paths import PROJECT_ROOT; print(PROJECT_ROOT)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(PROJECT_ROOT)


def test_production_modules_have_canonical_imports_and_snake_case_names():
    for path in (PROJECT_ROOT / "src/classroom_ai").rglob("*.py"):
        assert re.fullmatch(r"[a-z_][a-z0-9_]*\.py", path.name)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("src.", "kaggle_tools"))


def test_migration_targets_exist_and_artifact_bytes_unchanged():
    manifest = PROJECT_ROOT / "docs/architecture/path_migration_20261015.json"
    for record in json.loads(manifest.read_text(encoding="utf-8")):
        path = PROJECT_ROOT / record["new"]
        assert path.is_file(), record
        if path.suffix not in (".py", ".md", ".txt", ".in", ".ipynb", ".yaml", ".toml"):
            with path.open("rb") as stream:
                assert (
                    hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256_before"]
                ), record


def test_legacy_video_cli_delegates_to_canonical_main():
    from src.video.__main__ import main as legacy_main

    from classroom_ai.video.__main__ import main

    assert legacy_main is main


def test_terminal_timer_never_queries_widgets_after_shutdown():
    from classroom_ai.interfaces.terminal import VideoTerminal

    job = Mock()
    job.read_updates.return_value = []
    app = VideoTerminal(job=job)
    app.query_one = Mock(side_effect=AssertionError("DOM queried after shutdown"))
    app.poll_job()
    app.query_one.assert_not_called()
    job.read_updates.assert_not_called()
