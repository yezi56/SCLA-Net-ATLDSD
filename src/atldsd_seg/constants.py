"""Project constants for ATLDSD semantic segmentation."""

CLASS_NAMES = [
    "background",
    "leaf",
    "rust",
    "alternaria_leaf_spot",
    "gray_spot",
    "brown_spot",
]

NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255

LEAF_CLASS_ID = 1
LESION_CLASS_IDS = (2, 3, 4, 5)

SEVERITY_BINS = (
    ("healthy_or_trace", 0.00, 0.01),
    ("mild", 0.01, 0.05),
    ("moderate", 0.05, 0.15),
    ("severe", 0.15, 1.01),
)
