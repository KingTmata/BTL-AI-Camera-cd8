import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "datasets"))
try:
    import label_studio_to_yolo as converter
except ModuleNotFoundError:
    converter = None


class LabelStudioToYoloTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(converter, "The Label Studio converter has not been implemented")
        scratch_root = ROOT / ".temp"
        scratch_root.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch_root)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.train_hash, self.val_hash = "a" * 64, "b" * 64
        self.frames = []
        self.tasks = []
        for index, source in enumerate((self.train_hash, self.val_hash)):
            image_id = f"video_{source[:16]}_frame_00000{index}"
            image = self.images / f"{image_id}.jpg"
            Image.new("RGB", (100, 80), color=(index * 20, 0, 0)).save(image)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            frame = {
                "image_id": image_id,
                "filename": image.name,
                "image_sha256": digest,
                "source_video_sha256": source,
                "source_frame_index": index,
                "source_pts": index * 3,
                "time_base": "1/30",
                "timestamp_relative_seconds": index / 10,
                "width": 100,
                "height": 80,
            }
            rect = {
                "id": f"person{index}",
                "type": "rectanglelabels",
                "original_width": 100,
                "original_height": 80,
                "image_rotation": 0,
                "value": {
                    "x": 10,
                    "y": 20,
                    "width": 30,
                    "height": 40,
                    "rotation": 0,
                    "rectanglelabels": ["person"],
                },
            }
            point = {
                "id": f"head{index}",
                "parentID": rect["id"],
                "type": "keypointlabels",
                "original_width": 100,
                "original_height": 80,
                "image_rotation": 0,
                "value": {"x": 25, "y": 25, "keypointlabels": ["head"]},
            }
            self.frames.append(frame)
            self.tasks.append(
                {
                    "id": index + 1,
                    "data": dict(frame),
                    "annotations": [
                        {"id": index + 10, "was_cancelled": False, "result": [rect, point]}
                    ],
                    "predictions": [{"result": [rect]}],
                }
            )
        self.manifest = {"frames": self.frames}
        self.review = {
            "all_807_user_review_confirmed": True,
            "approved_image_ids": [r["image_id"] for r in self.frames],
        }
        self.split = {"train_video_sha256": self.train_hash, "val_video_sha256": self.val_hash}

    def build(self, task="detect"):
        return converter.convert_dataset(
            self.tasks,
            self.manifest,
            self.review,
            self.split,
            self.images,
            self.root / "dataset",
            task=task,
        )

    def test_detect_uses_approved_annotations_and_correct_normalized_box(self):
        report = self.build()
        self.assertEqual(report["counts"], {"train": 1, "val": 1})
        label = self.root / "dataset" / "labels" / "train" / (self.frames[0]["image_id"] + ".txt")
        values = [float(x) for x in label.read_text().split()]
        self.assertEqual(values, [0, 0.25, 0.4, 0.3, 0.4])
        self.assertTrue(
            (self.root / "dataset" / "images" / "val" / self.frames[1]["filename"]).is_file()
        )

    def test_prediction_only_or_missing_review_does_not_become_training_label(self):
        self.tasks[0]["annotations"] = []
        with self.assertRaisesRegex(ValueError, "annotation"):
            self.build()
        self.assertFalse((self.root / "dataset").exists())
        self.tasks[0]["annotations"] = [{"id": 1, "result": []}]
        self.review["approved_image_ids"].remove(self.frames[0]["image_id"])
        with self.assertRaisesRegex(ValueError, "review|duyệt"):
            self.build()

    def test_rejects_corrupt_images_and_invalid_boxes_without_partial_output(self):
        self.tasks[0]["annotations"][0]["result"][0]["value"]["width"] = 150
        with self.assertRaisesRegex(ValueError, "box|bbox"):
            self.build()
        self.assertFalse((self.root / "dataset").exists())
        self.tasks[0]["annotations"][0]["result"][0]["value"]["width"] = 30
        image = self.images / self.frames[0]["filename"]
        image.write_bytes(image.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash|SHA"):
            self.build()
        self.assertFalse((self.root / "dataset").exists())

    def test_rejects_leakage_and_duplicate_or_ambiguous_annotations(self):
        self.split["val_video_sha256"] = self.train_hash
        with self.assertRaisesRegex(ValueError, "split|video"):
            self.build()
        self.split["val_video_sha256"] = self.val_hash
        self.tasks[0]["annotations"].append(dict(self.tasks[0]["annotations"][0]))
        with self.assertRaisesRegex(ValueError, "annotation"):
            self.build()

    def test_pose_requires_explicit_visibility_and_preserves_13_point_order(self):
        with self.assertRaisesRegex(ValueError, "visibility"):
            self.build(task="pose13")
        for task in self.tasks:
            task["annotations"][0]["result"][1]["value"]["visibility"] = 2
        report = self.build(task="pose13")
        self.assertEqual(report["counts"]["train"], 1)
        label = self.root / "dataset" / "labels" / "train" / (self.frames[0]["image_id"] + ".txt")
        values = [float(x) for x in label.read_text().split()]
        self.assertEqual(len(values), 44)
        self.assertEqual(values[5:8], [0.25, 0.25, 2])
        self.assertEqual(values[8:], [0] * 36)

    def test_rejects_orphan_pose_points(self):
        for task in self.tasks:
            task["annotations"][0]["result"][1]["value"]["visibility"] = 2
        self.tasks[0]["annotations"][0]["result"][1]["parentID"] = "wrong-person"
        with self.assertRaisesRegex(ValueError, "parent"):
            self.build(task="pose13")


if __name__ == "__main__":
    unittest.main()
