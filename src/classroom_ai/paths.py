"""Locate external project assets without coupling paths to module nesting depth."""

import os
from pathlib import Path


def find_project_root():
    configured = os.environ.get("CLASSROOM_AI_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for directory in Path(__file__).resolve().parents:
        if (directory / "pyproject.toml").is_file() and (directory / "configs").is_dir():
            return directory
    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()
