"""Project8 v0.4 names and the immutable v0.3 class contract."""

LEGACY_PROJECT_NAMES = ("person", "table", "chair", "laptop", "cell phone", "backpack", "book", "cup")
PROJECT_NAMES = LEGACY_PROJECT_NAMES[:-1] + ("bottle",)
COCO_TO_PROJECT = {0: 0, 60: 1, 56: 2, 63: 3, 67: 4, 24: 5, 73: 6, 41: 7}
COCO_NAMES = {0: "person", 60: "dining table", 56: "chair", 63: "laptop", 67: "cell phone",
              24: "backpack", 73: "book", 41: "cup"}
