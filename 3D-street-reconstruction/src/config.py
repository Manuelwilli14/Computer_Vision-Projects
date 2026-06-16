"""
Global project settings: semantic classes, colors, default paths.
"""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

SAM_CHECKPOINT = CHECKPOINT_DIR / "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

# -----------------------------------------------------------------------
# Cityscapes Classes (label_id -> name)
# These are the native classes of DeepLabV3 trained on Cityscapes (19 classes)
# -----------------------------------------------------------------------
CITYSCAPES_CLASSES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic_light",
    7: "traffic_sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "cyclist",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

# -----------------------------------------------------------------------
# Grouping of the 19 Cityscapes classes into 6 supercategories used for the 3D visualization required by the project
# -----------------------------------------------------------------------
SUPER_CATEGORIES = {
    "road": ["road", "sidewalk", "terrain"],
    "vehicle": ["car", "truck", "bus", "train", "motorcycle", "bicycle", "cyclist"],
    "building": ["building", "wall", "fence"],
    "vegetation": ["vegetation"],
    "sky": ["sky"],
    "other": ["pole", "traffic_light", "traffic_sign", "person"],
}

# RGB color (0-255) associated with each supercategory
SUPER_CATEGORY_COLORS = {
    "road": (128, 64, 128),
    "vehicle": (0, 0, 142),
    "building": (70, 70, 70),
    "vegetation": (107, 142, 35),
    "sky": (70, 130, 180),
    "other": (0, 0, 0),
}

# Construct a direct mapping: label_id Cityscapes (0-18) -> supercategory
# Associate each Cityscapes label_id with its corresponding supercategory
LABEL_TO_SUPERCAT = {}
for super_cat, members in SUPER_CATEGORIES.items():
    for class_name, class_id in [(v, k) for k, v in CITYSCAPES_CLASSES.items()]:
        if class_name in members:
            LABEL_TO_SUPERCAT[class_id] = super_cat



# Default camera parameters (to be adjusted according to your RGB-D sensor)
# Example based on a Kinect / RealSense type camera
DEFAULT_INTRINSICS = {
    "width": 640,
    "height": 480,
    "fx": 525.0,
    "fy": 525.0,
    "cx": 319.5,
    "cy": 239.5,
}

# Scale factor for depth (depth_raw / DEPTH_SCALE = depth in meters)
DEPTH_SCALE = 1000.0
DEPTH_TRUNC = 6.0  # truncate depth beyond 6 meters (filters noise)
