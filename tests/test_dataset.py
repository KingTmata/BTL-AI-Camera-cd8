import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.dataset import NAMES, assign_splits, groups, inventory, labels, lock_release, sha256, validate, verify_release


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for index in range(8):
            image = self.root / f"image{index}.png"
            label = image.with_suffix(".txt")
            Image.new("RGB", (20, 20), (index*25, 0, 0)).save(image)
            label.write_text("\n".join(f"{i} 0.5 0.5 0.2 0.2" for i in range(len(NAMES))))
            self.rows.append({"image_id": f"img{index}", "image_path": image.name, "label_path": label.name,
                              "source_id": "camera_a", "session_id": f"session{index//2}",
                              "split_group_id": f"group{index//2}", "split": "",
                              "annotation_status": "approved", "annotator": "A", "reviewer": "B",
                              "review_status": "approved", "license_or_consent": "group consent",
                              "near_duplicate_status": "approved", "needs_review": "no"})

    def test_inventory_pending_and_orphan_label_rejected(self):
        rows = inventory(self.root, self.root, "member_a", root=self.root)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r["annotation_status"] == "unlabeled" for r in rows))
        (self.root / "orphan.txt").write_text("")
        with self.assertRaises(ValueError):
            inventory(self.root, self.root, "member_a", root=self.root)

    def test_negative_requires_independent_confirmation(self):
        rows = assign_splits(self.rows)
        row = rows[0]
        (self.root / row["label_path"]).write_text("")
        row["review_status"] = "pending"
        self.assertFalse(validate(rows, self.root, release=True)[0]["valid"])
        row["review_status"] = "approved"
        self.assertTrue(validate(rows, self.root, release=True)[0]["valid"])

    def test_invalid_geometry_class_and_duplicate_box(self):
        path = self.root / "invalid.txt"
        for text in ("8 .5 .5 .2 .2", "1 .9 .5 .4 .2", "1 nan .5 .2 .2", "1.0 .5 .5 .2 .2",
                     "1 .5 .5 .2 .2\n1 .5 .5 .2 .2"):
            path.write_text(text)
            with self.assertRaises(ValueError):
                labels(path)

    def test_missing_label_corrupt_image_exact_duplicate_and_escape(self):
        rows = [dict(r) for r in self.rows]
        rows[0]["label_path"] = "missing.txt"
        rows[1]["image_path"] = "../outside.jpg"
        (self.root / "image2.png").write_bytes(b"broken")
        (self.root / "image4.png").write_bytes((self.root / "image3.png").read_bytes())
        report, _ = validate(rows, self.root)
        self.assertFalse(report["valid"])
        self.assertGreaterEqual(len(report["errors"]), 4)

    def test_transitive_groups_and_reproducible_split(self):
        self.rows[2]["duplicate_group_id"] = "related"
        self.rows[4]["duplicate_group_id"] = "related"
        assigned = assign_splits(self.rows)
        self.assertEqual(assigned, assign_splits(self.rows))
        for group in groups(assigned):
            self.assertEqual(len({r["split"] for r in group}), 1)
        self.assertEqual(len(groups(assigned)), 3)

    def test_source_split_conflict_and_leakage(self):
        self.rows[0]["original_split"] = "train"
        self.rows[1]["original_split"] = "val"
        with self.assertRaises(ValueError):
            assign_splits(self.rows)
        self.rows[0]["split"] = "train"
        self.rows[1]["split"] = "val"
        self.assertFalse(validate(self.rows, self.root)[0]["valid"])

    def test_release_requires_review_and_permission(self):
        rows = assign_splits(self.rows)
        rows[0]["reviewer"] = "A"
        rows[1]["near_duplicate_status"] = "pending"
        rows[2]["license_or_consent"] = ""
        self.assertGreaterEqual(len(validate(rows, self.root, release=True)[0]["errors"]), 3)

    def test_lock_and_checksum_detect_modified_added_and_missing_files(self):
        destination = self.root / "dataset/v1"
        lock_release(assign_splits(self.rows), destination, self.root)
        self.assertEqual(len(verify_release(destination)), 64)
        with self.assertRaises(ValueError):
            lock_release(assign_splits(self.rows), destination, self.root)
        extra = destination / "extra.txt"
        extra.write_text("extra")
        with self.assertRaises(ValueError):
            verify_release(destination)
        extra.unlink()
        label = next((destination / "labels").rglob("*.txt"))
        original = label.read_bytes()
        label.write_text("")
        with self.assertRaises(ValueError):
            verify_release(destination)
        label.write_bytes(original)
        label.unlink()
        with self.assertRaises(ValueError):
            verify_release(destination)

    def test_in_place_snapshot_ignores_only_known_trainer_caches(self):
        destination = self.root / "dataset/primary"
        lock_release(assign_splits(self.rows), destination, self.root)
        marker = destination / "RELEASE.json"
        marker.write_text(json.dumps({"schema_version": 1, "storage": "in_place", "status": "locked"}))
        checksums = destination / "checksums.json"
        entries = json.loads(checksums.read_text())
        entries["RELEASE.json"] = sha256(marker)
        checksums.write_text(json.dumps(entries))
        original_digest = verify_release(destination)

        cache = destination / "labels/train.cache"
        cache.write_bytes(b"trainer cache")
        self.assertEqual(verify_release(destination), original_digest)
        cache.write_bytes(b"regenerated cache")
        self.assertEqual(verify_release(destination), original_digest)

        unknown = destination / "labels/extra.cache"
        unknown.write_bytes(b"unlisted file")
        with self.assertRaises(ValueError):
            verify_release(destination)
        unknown.unlink()
        label = next((destination / "labels").rglob("*.txt"))
        label.write_text("")
        with self.assertRaises(ValueError):
            verify_release(destination)

    def test_copied_release_does_not_ignore_trainer_cache(self):
        destination = self.root / "dataset/v1"
        lock_release(assign_splits(self.rows), destination, self.root)
        (destination / "labels/train.cache").write_bytes(b"new file")
        with self.assertRaises(ValueError):
            verify_release(destination)


if __name__ == "__main__":
    unittest.main()
