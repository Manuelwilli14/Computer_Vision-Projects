"""
2D semantic segmentation of street images.

Combines two approaches:
- DeepLabV3 (torchvision, pre-trained on Cityscapes-like data via COCO/VOC->fallback) to obtain CLASS labels (road, car, building, vegetation...)
- Segment Anything (SAM) to refine the edgess / propose precise masks that can be labeled using the output from DeepLabV3 (majority voting).

Output: a mask (H, W) where each pixel contains the supercategory ID
(0..N-1, see config.SUPER_CATEGORIES) as well as a colored visualization image.

"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
import cv2

from config import (
    CITYSCAPES_CLASSES,
    SUPER_CATEGORIES,
    SUPER_CATEGORY_COLORS,
    LABEL_TO_SUPERCAT,
    SAM_CHECKPOINT,
    SAM_MODEL_TYPE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sorted list of supercategories (index = ID used in the form)
SUPER_CAT_LIST = list(SUPER_CATEGORIES.keys())
SUPER_CAT_TO_ID = {name: i for i, name in enumerate(SUPER_CAT_LIST)}


# DeepLabV3 - Dense Semantic Segmentation
def load_deeplab_model():
    """
    Load a pre-trained DeepLabV3 model. 
    By default, torchvision provides weights trained on COCO-with-VOC-labels (21 classes). 
    For real-world “on-the-street” use, it is recommended to replace these weights 
    with a model fine-tuned on Cityscapes (19 classes), loaded via torch.hub or a local checkpoint.

    Here, we illustrate the architecture using the default weights and provide an adaptable class 
    mapping (see map_voc_to_supercat).
    """
    weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
    model.eval().to(DEVICE)
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    return model, preprocess, categories


VOC_TO_SUPERCAT = {
    "car": "vehicle",
    "bus": "vehicle",
    "train": "vehicle",
    "motorbike": "vehicle",
    "bicycle": "vehicle",
    "person": "human",
    "pottedplant": "vegetation",
    "tvmonitor": "object",
    "background": "sky",  # fallback grossier
}


def run_deeplab(model, preprocess, categories, image_pil):
    #Returns a (H, W) mask of supercategories (str) from DeepLabV3
    batch = preprocess(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(batch)["out"][0]
    class_map = output.argmax(0).cpu().numpy()  # (H, W) class ids VOC/COCO

    h, w = class_map.shape
    supercat_map = np.full((h, w), SUPER_CAT_TO_ID["other"], dtype=np.uint8)

    for class_id in np.unique(class_map):
        class_name = categories[class_id]
        super_cat = VOC_TO_SUPERCAT.get(class_name, "other")
        # Hypothesis: large, low-lying “background” areas = road; at the top = sky
        supercat_map[class_map == class_id] = SUPER_CAT_TO_ID[super_cat]

    # Simple heuristic to distinguish road / building / sky in the "background"
    # (useful with default weights that don't know these classes) :
    bg_mask = class_map == categories.index("__background__") if "__background__" in categories else (class_map == 0)
    supercat_map = _refine_background(image_pil, bg_mask, supercat_map)

    return supercat_map


def _refine_background(image_pil, bg_mask, supercat_map):
    """
    Fallback heuristic (without a Cityscapes model): separates the background
    into sky, road, and building based on vertical position and texture.
    Replace this function with direct inference from a Cityscapes model
    for a robust result.
    """
    h, w = bg_mask.shape
    img = np.array(image_pil.resize((w, h)))

    for y in range(h):
        row_mask = bg_mask[y, :]
        if not row_mask.any():
            continue
        relative_y = y / h
        if relative_y < 0.35:
            cat = "sky"
        elif relative_y < 0.65:
            cat = "building"
        else:
            cat = "road"
        supercat_map[y, row_mask] = SUPER_CAT_TO_ID[cat]

    return supercat_map



#Segment Anything Model (SAM) 
def load_sam_model():
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint SAM introuvable : {SAM_CHECKPOINT}\n"
            "Téléchargez-le avec :\n"
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
    the majority supercategory from DeepLabV3. This results in
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

    model, preprocess, categories = load_deeplab_model()
    supercat_map = run_deeplab(model, preprocess, categories, image_pil)

    if use_sam:
        try:
            mask_generator = load_sam_model()
            image_rgb = np.array(image_pil)
            supercat_map = refine_with_sam(mask_generator, image_rgb, supercat_map)
        except FileNotFoundError as e:
            print(f"[SAM] {e}\n[SAM] Étape SAM ignorée, utilisation de DeepLabV3 seul.")

    color_map = colorize_supercat_map(supercat_map)

    if output_path:
        cv2.imwrite(str(output_path), cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))
        label_path = Path(output_path).with_suffix(".labels.npy")
        np.save(label_path, supercat_map)
        print(f"Masque coloré sauvegardé : {output_path}")
        print(f"Labels bruts sauvegardés : {label_path}")

    return supercat_map, color_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation sémantique d'une image de rue")
    parser.add_argument("--image", required=True, help="Chemin vers l'image d'entrée")
    parser.add_argument("--output", required=True, help="Chemin du masque coloré de sortie (.png)")
    parser.add_argument("--no-sam", action="store_true", help="Désactive le raffinement SAM")
    args = parser.parse_args()

    segment_image(args.image, args.output, use_sam=not args.no_sam)
