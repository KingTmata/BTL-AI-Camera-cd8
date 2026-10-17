"""Small, portable integrity helpers for the Project8 experiment campaign."""

import hashlib
import json
import math
import os
import zipfile
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_epoch(rows, expected_epochs):
    """CSV is one based; reject incomplete/nonfinite history before choosing AP50."""
    if expected_epochs < 1 or len(rows) != expected_epochs:
        raise ValueError("Incomplete epoch history")
    parsed = []
    for index, row in enumerate(rows, 1):
        epoch = float(row["epoch"])
        ap50 = float(row["metrics/mAP50(B)"])
        ap = float(row["metrics/mAP50-95(B)"])
        if not all(math.isfinite(value) for value in (epoch, ap50, ap)):
            raise ValueError("Nonfinite epoch metric")
        if epoch != index or not 0 <= ap50 <= 1 or not 0 <= ap <= 1:
            raise ValueError("Invalid epoch sequence or AP range")
        parsed.append((ap50, ap, -index))
    return -max(parsed)[2]


def rank_results(results):
    """Only rank measured results using identical ground truth and protocol."""
    if not results:
        return []
    first = next(iter(results.values()))
    identity = (first["manifest_sha256"], first["protocol"])
    for record in results.values():
        if (record["manifest_sha256"], record["protocol"]) != identity:
            raise ValueError("Cannot rank different datasets or protocols together")
        value = float(record["mAP50"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid AP50")
    return sorted(results, key=lambda name: (-results[name]["mAP50"], name))


def verified_bundle(root, sources, destination):
    """Atomically publish a ZIP after CRC and every payload hash are verified."""
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    payload = {}
    for source in sources:
        source = Path(source)
        if source.is_symlink():
            raise ValueError("Symlinks are not campaign payload")
        path = source.resolve()
        if not path.is_relative_to(root) or path == destination:
            raise ValueError("Payload outside run directory or recursive archive")
        name = path.relative_to(root).as_posix()
        if name == "FILES.json" or name in payload:
            raise ValueError("Duplicate or reserved payload name")
        if not path.is_file():
            raise ValueError("Missing payload")
        payload[name] = path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    manifest = {
        name: {"sha256": file_sha(path), "bytes": path.stat().st_size}
        for name, path in payload.items()
    }
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in payload.items():
            archive.write(path, name)
        archive.writestr("FILES.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt archive")
        for name, metadata in manifest.items():
            digest = hashlib.sha256()
            total = 0
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    total += len(block)
                    digest.update(block)
            if digest.hexdigest() != metadata["sha256"] or total != metadata["bytes"]:
                raise ValueError("Payload changed while building archive")
    os.replace(temporary, destination)
    return file_sha(destination)
