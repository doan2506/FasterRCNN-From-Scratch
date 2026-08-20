import os
import json
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
    """

    def __init__(self, annotation_file: str, image_dir: str, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

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

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_info[img_id]

        # Determine full image path flexibly
        file_name = img_info.get("file_name", img_id)
        image_path = os.path.join(self.image_dir, os.path.basename(file_name))
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, file_name)

        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        # Parse ground-truth boxes and labels
        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []

        for ann in anns:
            cls_name = ann["class"]
            if cls_name in self.class_to_idx:
                label = self.class_to_idx[cls_name]
                bbox = ann["bbox"]  # [xmin, ymin, xmax, ymax]
                boxes.append(bbox)
                labels.append(label)

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        if self.transforms is not None:
            image_tensor, boxes_tensor, labels_tensor, orig_shape = self.transforms(
                image, boxes_tensor, labels_tensor
            )
        else:
            image_tensor = TF.to_tensor(image)
            orig_shape = (orig_h, orig_w)

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
