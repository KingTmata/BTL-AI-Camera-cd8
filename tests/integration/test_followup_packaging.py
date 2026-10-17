import ast
import base64
import hashlib
import json
import re
import tempfile
import unittest
import zlib
from pathlib import Path

from notebooks.kaggle.build_project8_followups import JOBS, build_followups


class FollowupPackagingTests(unittest.TestCase):
    def fixture(self, root):
        for config in JOBS.values():
            for relative in config["sources"]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("def portable_helper():\n    return 42\n", encoding="utf-8")
        path = root / "src/classroom_ai/experiments/validation_replay.py"
        path.write_text(
            "from classroom_ai.inference.variants import merge_detections\n" + path.read_text()
        )
        (root / "reports/training").mkdir(parents=True)
        (root / "reports/training/project8_v05_log.md").write_text(
            "# Existing results\n", encoding="utf-8"
        )

    def test_packages_current_sources_without_deleted_logs_or_archives(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            records = build_followups(root)
            self.assertEqual(set(records), set(JOBS))
            for name, record in records.items():
                cell = (root / record["cell"]).read_text()
                packed = re.search(r"base64\.b64decode\('([^']+)'\)", cell).group(1)
                source = zlib.decompress(base64.b64decode(packed)).decode()
                self.assertEqual(
                    hashlib.sha256(source.encode()).hexdigest(), record["embedded_source_sha256"]
                )
                ast.parse(cell)
                ast.parse(source)
                self.assertNotIn("from classroom_ai.experiments.", source)
                self.assertIn("PREPARED['dataset_manifest_sha256'] == MANIFEST_SHA256", source)
                notebook = json.loads((root / record["notebook"]).read_text())
                self.assertEqual("".join(notebook["cells"][1]["source"]), cell)
                self.assertEqual(notebook["cells"][1]["outputs"], [])
                self.assertIsNone(notebook["cells"][1]["execution_count"])
                self.assertEqual(record["execution_status"], "prepared_not_GPU_executed")
            self.assertFalse((root / "reports/campaign_20261010").exists())
            self.assertFalse(list(root.rglob("*.zip")))
            self.assertFalse(list(root.rglob("*.json")))
            self.assertTrue(
                (root / "reports/training/project8_v05_log.md")
                .read_text()
                .startswith("# Existing results\n")
            )

    def test_malformed_source_stops_before_any_package_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            broken = root / JOBS["source_ablations_train"]["sources"][-1]
            broken.write_text("def broken(:\n")
            with self.assertRaises(SyntaxError):
                build_followups(root)
            self.assertFalse(list(root.rglob("*_cell_20261010.py")))
            self.assertFalse(list(root.rglob("*.ipynb")))
            self.assertEqual(
                (root / "reports/training/project8_v05_log.md").read_text(), "# Existing results\n"
            )


if __name__ == "__main__":
    unittest.main()
