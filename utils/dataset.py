import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from utils.augmentations import DetectionTransforms


CLASSES = ["bottle", "cup", "chair", "laptop", "backpack"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}


class ObjectDetectionDataset(Dataset):
    """
    Dataset loader for Object Detection using JSON annotations.
    Supports Select-Mosaic 4-image data augmentation with criteria-based candidate selection.
    """

    def __init__(self, annotation_file: str, image_dir: str, transforms=None, use_mosaic=False, mosaic_prob=0.3):
        self.image_dir = image_dir
        self.transforms = transforms
        self.use_mosaic = use_mosaic
        self.mosaic_prob = mosaic_prob

        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.classes = data.get("classes", CLASSES)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(self.classes)}
        self.images_info = {img["id"]: img for img in data.get("images", [])}

        # Group annotations by image_id
        self.annotations = {}
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.image_ids = list(self.images_info.keys())

        # Pre-index dataset for Select-Mosaic criteria-based matching
        self.class_to_indices = {cls_name: [] for cls_name in self.classes}
        self.small_obj_indices = []
        self.hard_class_indices = []

        for idx, img_id in enumerate(self.image_ids):
            anns = self.annotations.get(img_id, [])
            img_classes = set()
            has_small = False
            has_hard = False
            for a in anns:
                c = a.get("class")
                if c in self.class_to_indices:
                    img_classes.add(c)
                bbox = a.get("bbox", [0, 0, 0, 0])
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if (w * h) < (64 * 64):
                    has_small = True
                if c in ["backpack", "chair", "bottle"]:
                    has_hard = True

            for c in img_classes:
                self.class_to_indices[c].append(idx)
            if has_small:
                self.small_obj_indices.append(idx)
            if has_hard:
                self.hard_class_indices.append(idx)

    def __len__(self):
        return len(self.image_ids)

    def _load_image_and_boxes(self, idx: int):
        """
        Loads a single PIL image and its corresponding ground-truth boxes and labels.
        """
        img_id = self.image_ids[idx]
        img_info = self.images_info[img_id]

        file_name = img_info.get("file_name", img_id)
        image_path = os.path.join(self.image_dir, os.path.basename(file_name))
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, file_name)

        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []

        for ann in anns:
            cls_name = ann["class"]
            if cls_name in self.class_to_idx:
                boxes.append(ann["bbox"])
                labels.append(self.class_to_idx[cls_name])

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        return image, boxes_tensor, labels_tensor, img_id, (orig_w, orig_h)

    def _select_mosaic_candidates(self, base_idx: int) -> list:
        """
        [STEP 1: Selection Strategy for Select-Mosaic]
        Selects 3 targeted partner images based on:
        1. Class diversity: Picks images containing classes absent in the base image.
        2. Hard-class focus: Picks images containing difficult/bottleneck classes (backpack, chair, bottle).
        3. Scale enrichment: Picks images containing small objects (< 64x64px).
        """
        base_id = self.image_ids[base_idx]
        base_anns = self.annotations.get(base_id, [])
        base_classes = {a.get("class") for a in base_anns}
        missing_classes = [c for c in self.classes if c not in base_classes]

        selected_indices = []

        # Criterion 1: Complementary class selection (diversity)
        if missing_classes:
            target_cls = random.choice(missing_classes)
            pool = self.class_to_indices.get(target_cls, [])
            if pool:
                cand = random.choice(pool)
                if cand != base_idx:
                    selected_indices.append(cand)

        # Criterion 2: Hard class selection (backpack, chair, bottle)
        if len(selected_indices) < 2 and self.hard_class_indices:
            cand = random.choice(self.hard_class_indices)
            if cand != base_idx and cand not in selected_indices:
                selected_indices.append(cand)

        # Criterion 3: Small object scale enrichment (< 64x64)
        if len(selected_indices) < 3 and self.small_obj_indices:
            cand = random.choice(self.small_obj_indices)
            if cand != base_idx and cand not in selected_indices:
                selected_indices.append(cand)

        # Fallback filler if any pool was empty or duplicate
        while len(selected_indices) < 3:
            rand_idx = random.randint(0, len(self.image_ids) - 1)
            if rand_idx != base_idx and rand_idx not in selected_indices:
                selected_indices.append(rand_idx)

        return selected_indices

    def _load_mosaic(self, idx: int):
        """
        [STEPS 2 & 3: Select-Mosaic 4-Image Stitching & Target Transformation]
        1. Calls _select_mosaic_candidates(idx) to pick 3 targeted partner images.
        2. Stitches 4 selected images onto a 2x2 grid with random center point (xc, yc).
        3. Scales, shifts, clips and filters corresponding bounding boxes and labels.
        """
        target_size = 640
        if self.transforms is not None and hasattr(self.transforms, "target_size"):
            target_size = self.transforms.target_size[0]

        S = target_size
        xc = random.randint(int(0.35 * S), int(0.65 * S))
        yc = random.randint(int(0.35 * S), int(0.65 * S))

        # Quadrants: (offset_x, offset_y, quadrant_width, quadrant_height)
        quadrants = [
            (0, 0, xc, yc),           # Top-Left (base image)
            (xc, 0, S - xc, yc),      # Top-Right (selected partner 1)
            (0, yc, xc, S - yc),      # Bottom-Left (selected partner 2)
            (xc, yc, S - xc, S - yc)  # Bottom-Right (selected partner 3)
        ]

        # Step 1: Select 3 targeted candidate images
        partner_indices = self._select_mosaic_candidates(idx)
        mosaic_indices = [idx] + partner_indices

        # Step 2: Create canvas and paste quadrants
        canvas = Image.new("RGB", (S, S), (114, 114, 114))
        all_boxes = []
        all_labels = []

        for (off_x, off_y, qw, qh), m_idx in zip(quadrants, mosaic_indices):
            m_img, m_boxes, m_labels, _, (m_w, m_h) = self._load_image_and_boxes(m_idx)

            # Resize to quadrant dimensions and paste
            resized_img = m_img.resize((qw, qh))
            canvas.paste(resized_img, (off_x, off_y))

            # Step 3: Transform bounding boxes to composite coordinates
            if len(m_boxes) > 0:
                sx = qw / float(m_w)
                sy = qh / float(m_h)

                b = m_boxes.clone()
                b[:, 0] = (b[:, 0] * sx + off_x).clamp(min=0, max=S)
                b[:, 1] = (b[:, 1] * sy + off_y).clamp(min=0, max=S)
                b[:, 2] = (b[:, 2] * sx + off_x).clamp(min=0, max=S)
                b[:, 3] = (b[:, 3] * sy + off_y).clamp(min=0, max=S)

                # Filter valid bounding boxes (at least 2px width and height)
                valid = (b[:, 2] > b[:, 0] + 2) & (b[:, 3] > b[:, 1] + 2)
                if valid.any():
                    all_boxes.append(b[valid])
                    all_labels.append(m_labels[valid])

        if len(all_boxes) > 0:
            final_boxes = torch.cat(all_boxes, dim=0)
            final_labels = torch.cat(all_labels, dim=0)
        else:
            final_boxes = torch.zeros((0, 4), dtype=torch.float32)
            final_labels = torch.zeros((0,), dtype=torch.int64)

        return canvas, final_boxes, final_labels, (S, S)

    def __getitem__(self, idx):
        # 1. Apply Select-Mosaic Augmentation if enabled
        if self.use_mosaic and random.random() < self.mosaic_prob:
            image, boxes_tensor, labels_tensor, orig_shape = self._load_mosaic(idx)
            img_id = self.image_ids[idx]
        else:
            image, boxes_tensor, labels_tensor, img_id, (orig_w, orig_h) = self._load_image_and_boxes(idx)
            orig_shape = (orig_h, orig_w)

        # 2. Apply Photometric / Spatial transforms (Flip, Color Jitter, Resize, Normalize)
        if self.transforms is not None:
            image_tensor, boxes_tensor, labels_tensor, transformed_orig_shape = self.transforms(
                image, boxes_tensor, labels_tensor
            )
            if not (self.use_mosaic and random.random() < self.mosaic_prob):
                orig_shape = transformed_orig_shape
        else:
            image_tensor = TF.to_tensor(image)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": img_id,
            "orig_shape": orig_shape,
        }

        return image_tensor, target


def detection_collate_fn(batch):
    """
    Collate function to batch multiple images and bounding boxes.
    Supports multi-scale batches by padding all images in the batch to (max_h, max_w).
    """
    images, targets = zip(*batch)

    # Check if all images have the same shape
    heights = [img.shape[1] for img in images]
    widths = [img.shape[2] for img in images]

    max_h = max(heights)
    max_w = max(widths)

    if all(h == max_h for h in heights) and all(w == max_w for w in widths):
        stacked_images = torch.stack(images, dim=0)
    else:
        # Pad each image to (max_h, max_w) with 0
        padded_images = []
        for img in images:
            _, h, w = img.shape
            pad_h = max_h - h
            pad_w = max_w - w
            if pad_h > 0 or pad_w > 0:
                img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0)
            padded_images.append(img)
        stacked_images = torch.stack(padded_images, dim=0)

    return stacked_images, list(targets)
