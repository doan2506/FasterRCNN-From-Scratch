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
    Implements 4-Step Select-Mosaic Augmentation:
      1. Anchor/Base Image Selection
      2. Selective Retrieval (Small objects, Hard/Minority classes, Dense scenes)
      3. ROI-based Focused Crop & Paste
      4. Dynamic Center Point & Area-Retention Bounding Box Filtering
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

        # Pre-index dataset for Select-Mosaic Selective Retrieval (Step 2)
        self.class_to_indices = {cls_name: [] for cls_name in self.classes}
        self.small_obj_indices = []      # Images containing small objects (< 32x32px)
        self.hard_class_indices = []     # Images containing difficult/minority classes (backpack, chair, bottle)
        self.dense_scene_indices = []    # Images with high object density (>= 3 objects)

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
                w = max(0.0, bbox[2] - bbox[0])
                h = max(0.0, bbox[3] - bbox[1])
                # Small object threshold: area < 32x32 or dimension < 32px
                if (w * h) < (32 * 32) or w < 32 or h < 32:
                    has_small = True
                if c in ["backpack", "chair", "bottle"]:
                    has_hard = True

            for c in img_classes:
                self.class_to_indices[c].append(idx)
            if has_small:
                self.small_obj_indices.append(idx)
            if has_hard:
                self.hard_class_indices.append(idx)
            if len(anns) >= 3:
                self.dense_scene_indices.append(idx)

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

    def _selective_retrieval(self, base_idx: int) -> list:
        """
        [STEP 2: Selective Retrieval]
        Selects 3 companion images based on:
        1. Small Object Enrichment: Images containing small objects (< 32x32px).
        2. Hard/Minority Class Focus: Images containing bottleneck classes (backpack, chair, bottle).
        3. High Context Density / Multi-class Diversity: Dense scenes (>= 3 objects) or missing classes.
        """
        base_id = self.image_ids[base_idx]
        base_anns = self.annotations.get(base_id, [])
        base_classes = {a.get("class") for a in base_anns}
        missing_classes = [c for c in self.classes if c not in base_classes]

        selected_indices = []

        # 1. Select image with small objects
        if self.small_obj_indices:
            cand = random.choice(self.small_obj_indices)
            if cand != base_idx:
                selected_indices.append(cand)

        # 2. Select image with hard/minority classes (backpack, chair, bottle)
        if self.hard_class_indices:
            cand = random.choice(self.hard_class_indices)
            if cand != base_idx and cand not in selected_indices:
                selected_indices.append(cand)

        # 3. Select image with dense object count or missing class
        if missing_classes:
            target_cls = random.choice(missing_classes)
            pool = self.class_to_indices.get(target_cls, [])
            if pool:
                cand = random.choice(pool)
                if cand != base_idx and cand not in selected_indices:
                    selected_indices.append(cand)
        elif self.dense_scene_indices:
            cand = random.choice(self.dense_scene_indices)
            if cand != base_idx and cand not in selected_indices:
                selected_indices.append(cand)

        # Fallback to random if any pool candidate conflicted or was empty
        while len(selected_indices) < 3:
            rand_idx = random.randint(0, len(self.image_ids) - 1)
            if rand_idx != base_idx and rand_idx not in selected_indices:
                selected_indices.append(rand_idx)

        return selected_indices

    def _load_select_mosaic(self, base_idx: int):
        """
        [STEPS 1, 2, 3, 4: Full Select-Mosaic Implementation]
        1. Anchor Image: Base image at base_idx (Top-Left quadrant).
        2. Selective Retrieval: 3 companion images selected by criteria.
        3. ROI-based Focused Crop: Focuses on object clusters/small objects without squashing whole images.
        4. Dynamic Center Point & Area Retention Filter: Retains only boxes with >= 30% area inside crop.
        """
        target_size = 640
        if self.transforms is not None and hasattr(self.transforms, "target_size"):
            target_size = self.transforms.target_size[0]

        S = target_size

        # Step 4: Dynamic Center Pivot (avoiding rigid 50/50 division)
        xc = random.randint(int(0.35 * S), int(0.65 * S))
        yc = random.randint(int(0.35 * S), int(0.65 * S))

        # 4 Quadrants: (offset_x, offset_y, quadrant_width, quadrant_height)
        quadrants = [
            (0, 0, xc, yc),           # Top-Left: Anchor/Base image
            (xc, 0, S - xc, yc),      # Top-Right: Companion 1 (Small Object focused)
            (0, yc, xc, S - yc),      # Bottom-Left: Companion 2 (Hard Class focused)
            (xc, yc, S - xc, S - yc)  # Bottom-Right: Companion 3 (Dense Context focused)
        ]

        # Step 2: Retrieve 3 companion images
        companion_indices = self._selective_retrieval(base_idx)
        mosaic_indices = [base_idx] + companion_indices

        canvas = Image.new("RGB", (S, S), (114, 114, 114))
        all_boxes = []
        all_labels = []

        for (off_x, off_y, qw, qh), m_idx in zip(quadrants, mosaic_indices):
            m_img, m_boxes, m_labels, _, (w_img, h_img) = self._load_image_and_boxes(m_idx)

            # Step 3: ROI-based Crop around focused object cluster
            if len(m_boxes) > 0:
                # Select an anchor box (prioritize smaller objects or hard classes if available)
                target_box_idx = random.randint(0, len(m_boxes) - 1)
                tb = m_boxes[target_box_idx]
                tcx = (tb[0] + tb[2]) / 2.0
                tcy = (tb[1] + tb[3]) / 2.0

                # Crop window size (60% to 100% of original image dimensions)
                target_aspect = qw / float(qh)
                crop_scale = random.uniform(0.60, 1.0)

                if (w_img / float(h_img)) > target_aspect:
                    # Original image is "wider" than target aspect -> constrain by height first
                    ch = max(10, min(int(h_img * crop_scale), h_img))
                    cw = max(10, min(int(ch * target_aspect), w_img))
                else:
                    # Original image is "taller" than target aspect -> constrain by width first
                    cw = max(10, min(int(w_img * crop_scale), w_img))
                    ch = max(10, min(int(cw / target_aspect), h_img))

                cx1 = max(0, min(int(tcx - cw / 2.0), w_img - cw))
                cy1 = max(0, min(int(tcy - ch / 2.0), h_img - ch))
                cx2 = cx1 + cw
                cy2 = cy1 + ch

                cropped_img = m_img.crop((cx1, cy1, cx2, cy2))
                resized_img = cropped_img.resize((qw, qh))
                canvas.paste(resized_img, (off_x, off_y))

                # Step 4: Transform, clamp, and filter bounding boxes by Area Retention Ratio
                sx = qw / float(cw)
                sy = qh / float(ch)

                for b, l in zip(m_boxes, m_labels):
                    orig_area = (b[2] - b[0]) * (b[3] - b[1])
                    ix1 = max(b[0].item(), cx1) - cx1
                    iy1 = max(b[1].item(), cy1) - cy1
                    ix2 = min(b[2].item(), cx2) - cx1
                    iy2 = min(b[3].item(), cy2) - cy1

                    inter_w = max(0.0, ix2 - ix1)
                    inter_h = max(0.0, iy2 - iy1)
                    inter_area = inter_w * inter_h

                    # Filter condition: Box retains at least 30% area and is >= 3px
                    if orig_area > 0 and (inter_area / orig_area) >= 0.30 and inter_w >= 3.0 and inter_h >= 3.0:
                        fx1 = max(0.0, min(float(S), ix1 * sx + off_x))
                        fy1 = max(0.0, min(float(S), iy1 * sy + off_y))
                        fx2 = max(0.0, min(float(S), ix2 * sx + off_x))
                        fy2 = max(0.0, min(float(S), iy2 * sy + off_y))

                        if (fx2 > fx1 + 2.0) and (fy2 > fy1 + 2.0):
                            all_boxes.append(torch.tensor([[fx1, fy1, fx2, fy2]]))
                            all_labels.append(l.unsqueeze(0))
            else:
                # Background-only image: direct resize and paste
                resized_img = m_img.resize((qw, qh))
                canvas.paste(resized_img, (off_x, off_y))

        if len(all_boxes) > 0:
            final_boxes = torch.cat(all_boxes, dim=0)
            final_labels = torch.cat(all_labels, dim=0)
        else:
            final_boxes = torch.zeros((0, 4), dtype=torch.float32)
            final_labels = torch.zeros((0,), dtype=torch.int64)

        return canvas, final_boxes, final_labels, (S, S)

    def __getitem__(self, idx):
        # 1. Evaluate Select-Mosaic trigger condition exactly once
        is_mosaic = self.use_mosaic and (random.random() < self.mosaic_prob)

        if is_mosaic:
            image, boxes_tensor, labels_tensor, (orig_h, orig_w) = self._load_select_mosaic(idx)
            img_id = self.image_ids[idx]
        else:
            image, boxes_tensor, labels_tensor, img_id, (orig_w, orig_h) = self._load_image_and_boxes(idx)

        # 2. Apply Photometric / Spatial transforms (Flip, Color Jitter, Multi-scale Resize, Normalize)
        if self.transforms is not None:
            image_tensor, boxes_tensor, labels_tensor, (raw_orig_h, raw_orig_w) = self.transforms(
                image, boxes_tensor, labels_tensor
            )
            if not is_mosaic:
                orig_h, orig_w = raw_orig_h, raw_orig_w
        else:
            image_tensor = TF.to_tensor(image)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": img_id,
            "orig_shape": (orig_h, orig_w),
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
