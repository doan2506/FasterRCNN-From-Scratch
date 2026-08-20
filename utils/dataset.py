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
    Supports optional 4-image Selective Mosaic augmentation.
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

    def _load_mosaic(self, idx: int):
        """
        Selective Mosaic 4-Image Augmentation:
        Combines 4 random images into a 2x2 mosaic canvas with random center point (xc, yc).
        """
        target_size = 640
        if self.transforms is not None and hasattr(self.transforms, "target_size"):
            target_size = self.transforms.target_size[0]

        S = target_size
        xc = random.randint(int(0.35 * S), int(0.65 * S))
        yc = random.randint(int(0.35 * S), int(0.65 * S))

        # Quadrant placements: (offset_x, offset_y, quadrant_width, quadrant_height)
        quadrants = [
            (0, 0, xc, yc),           # Top-Left (current sample)
            (xc, 0, S - xc, yc),      # Top-Right
            (0, yc, xc, S - yc),      # Bottom-Left
            (xc, yc, S - xc, S - yc)  # Bottom-Right
        ]

        # Select 3 other random image indices
        other_indices = [random.randint(0, len(self.image_ids) - 1) for _ in range(3)]
        mosaic_indices = [idx] + other_indices

        canvas = Image.new("RGB", (S, S), (114, 114, 114))
        all_boxes = []
        all_labels = []

        for (off_x, off_y, qw, qh), m_idx in zip(quadrants, mosaic_indices):
            m_img, m_boxes, m_labels, _, (m_w, m_h) = self._load_image_and_boxes(m_idx)

            # Resize image to quadrant dimensions and paste
            resized_img = m_img.resize((qw, qh))
            canvas.paste(resized_img, (off_x, off_y))

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
        # 1. Apply Selective Mosaic Augmentation if enabled
        if self.use_mosaic and random.random() < self.mosaic_prob:
            image, boxes_tensor, labels_tensor, orig_shape = self._load_mosaic(idx)
            img_id = self.image_ids[idx]
        else:
            image, boxes_tensor, labels_tensor, img_id, (orig_w, orig_h) = self._load_image_and_boxes(idx)
            orig_shape = (orig_h, orig_w)

        # 2. Apply Spatial / Photometric transforms (Flip, Color Jitter, Resize, Normalize)
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
