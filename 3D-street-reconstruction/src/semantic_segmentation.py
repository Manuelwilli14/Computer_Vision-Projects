"""
2D semantic segmentation of street images.

Combines two approaches:
- SegFormer-B0 (Hugging Face transformers, fine-tuned on Cityscapes) to obtain the
  CLASS of each pixel among the 19 Cityscapes classes (road, car, building, vegetation...)
- Segment Anything (SAM, optional) to refine the edges: precise masks that are labeled
  using the output from SegFormer (majority voting).

Output: a mask (H, W) where each pixel contains the supercategory ID
(0..N-1, see config.SUPER_CATEGORIES) as well as a colored visualization image.

"""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from config import (
    CITYSCAPES_CLASSES,
    SUPER_CATEGORIES,
    SUPER_CATEGORY_COLORS,
    LABEL_TO_SUPERCAT,
    SAM_CHECKPOINT,
    SAM_MODEL_TYPE,
    SEGFORMER_MODEL,
    SEGFORMER_INPUT_PIXELS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sorted list of supercategories (index = ID used in the form)
SUPER_CAT_LIST = list(SUPER_CATEGORIES.keys())
SUPER_CAT_TO_ID = {name: i for i, name in enumerate(SUPER_CAT_LIST)}

# Lookup table: Cityscapes class ID (0-18) -> supercategory ID
CITYSCAPES_TO_SUPERCAT_ID = np.array(
    [SUPER_CAT_TO_ID[LABEL_TO_SUPERCAT[i]] for i in range(len(CITYSCAPES_CLASSES))], dtype=np.uint8
)


# SegFormer - Dense Semantic Segmentation (Cityscapes)
def _from_pretrained(cls):
    # Use the local Hugging Face cache if the model was already downloaded (no network
    # access at all), otherwise download it (first run only, ~15 MB)
    try:
        return cls.from_pretrained(SEGFORMER_MODEL, local_files_only=True)
    except OSError:
        return cls.from_pretrained(SEGFORMER_MODEL)


@lru_cache(maxsize=None)  # load once, reuse for every frame of a sequence
def load_segformer_model():
    """
    Load SegFormer fine-tuned on Cityscapes. Its 19 output classes are in the
    same order as config.CITYSCAPES_CLASSES, so they map directly to the
    supercategories through config.LABEL_TO_SUPERCAT.
    """
    processor = _from_pretrained(SegformerImageProcessor)
    model = _from_pretrained(SegformerForSemanticSegmentation).eval().to(DEVICE)
    if model.config.num_labels != len(CITYSCAPES_CLASSES):
        raise ValueError(
            f"{SEGFORMER_MODEL} predicts {model.config.num_labels} classes, "
            f"expected the {len(CITYSCAPES_CLASSES)} Cityscapes classes"
        )
    return model, processor


def run_segformer(model, processor, image_pil):
    #Returns a (H, W) mask of supercategory IDs from SegFormer
    w, h = image_pil.size
    # Resize to ~SEGFORMER_INPUT_PIXELS pixels keeping the aspect ratio
    # (the processor's default 512x512 would distort non-square images)
    scale = (SEGFORMER_INPUT_PIXELS / (w * h)) ** 0.5
    size = {"height": round(h * scale), "width": round(w * scale)}
    inputs = processor(images=image_pil, size=size, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits  # (1, 19, h/4, w/4) Cityscapes class scores
        # bring the scores back to the original resolution so that
        # label pixel (u, v) = image/depth pixel (u, v)
        logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
    class_map = logits.argmax(1)[0].cpu().numpy()  # (H, W) Cityscapes class ids 0-18

    return CITYSCAPES_TO_SUPERCAT_ID[class_map]



#Segment Anything Model (SAM)
@lru_cache(maxsize=None)
def load_sam_model():
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {SAM_CHECKPOINT}\n"
            "Download it with:\n"
            "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/"
        )

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,  # Decrease for speed, increase for greater accuracy
    )
    return mask_generator


def refine_with_sam(mask_generator, image_rgb, supercat_map):
    """
    Generate SAM (precise oversegmentation) masks and then assigns each one
    the majority supercategory from SegFormer. This results in
    much sharper contours while preserving class semantics.
    """
    masks = mask_generator.generate(image_rgb)
    refined = supercat_map.copy()

    for m in masks:
        seg = m["segmentation"]  # (H, W) bool
        if seg.sum() == 0:
            continue
        values, counts = np.unique(supercat_map[seg], return_counts=True)
        majority_label = values[np.argmax(counts)]
        refined[seg] = majority_label

    return refined


# Visualisation
def colorize_supercat_map(supercat_map):
    h, w = supercat_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for super_cat, cat_id in SUPER_CAT_TO_ID.items():
        color = SUPER_CATEGORY_COLORS[super_cat]
        color_img[supercat_map == cat_id] = color
    return color_img



def segment_image(image_path, output_path=None, use_sam=True):
    image_pil = Image.open(image_path).convert("RGB")

    model, processor = load_segformer_model()
    supercat_map = run_segformer(model, processor, image_pil)

    if use_sam:
        try:
            mask_generator = load_sam_model()
            image_rgb = np.array(image_pil)
            supercat_map = refine_with_sam(mask_generator, image_rgb, supercat_map)
        except FileNotFoundError as e:
            print(f"[SAM] {e}\n[SAM] SAM step skipped, using SegFormer only.")

    color_map = colorize_supercat_map(supercat_map)

    if output_path:
        cv2.imwrite(str(output_path), cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))
        label_path = Path(output_path).with_suffix(".labels.npy")
        np.save(label_path, supercat_map)
        print(f"Colored mask saved: {output_path}")
        print(f"Raw labels saved: {label_path}")

    return supercat_map, color_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic segmentation of a street image")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--output", required=True, help="Path to the output colored mask (.png)")
    parser.add_argument("--no-sam", action="store_true", help="Disable SAM refinement (faster)")
    args = parser.parse_args()

    segment_image(args.image, args.output, use_sam=not args.no_sam)
