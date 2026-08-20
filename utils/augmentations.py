import random
import torch
import torchvision.transforms.functional as F
from PIL import Image


# Multi-scale training sizes (randomly picked each sample)
MULTI_SCALE_SIZES = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]


class DetectionTransforms:
    """
    Data augmentation and transformation pipeline for Object Detection.
    Supports multi-scale training, random horizontal flip, color jitter,
    random expand+crop, and ImageNet normalization.
    """

    def __init__(self, target_size=(600, 600), is_train=True, multi_scale=True):
        self.target_size = target_size  # (height, width) — used as default / for val
        self.is_train = is_train
        self.multi_scale = multi_scale and is_train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor):
        """
        image: PIL Image
        boxes: Tensor of shape (N, 4) in [xmin, ymin, xmax, ymax]
        labels: Tensor of shape (N,)
        """
        raw_orig_w, raw_orig_h = image.size
        cur_w, cur_h = raw_orig_w, raw_orig_h

        if self.is_train and len(boxes) > 0:
            # 1. Random Color Jitter (50% probability - photometric distortion)
            if random.random() > 0.5:
                brightness = random.uniform(0.7, 1.3)
                contrast = random.uniform(0.7, 1.3)
                saturation = random.uniform(0.7, 1.3)
                hue = random.uniform(-0.05, 0.05)
                image = F.adjust_brightness(image, brightness)
                image = F.adjust_contrast(image, contrast)
                image = F.adjust_saturation(image, saturation)
                image = F.adjust_hue(image, hue)

            # 2. Random Expand + Crop (SSD-style, 40% probability)
            if random.random() > 0.6 and len(boxes) > 0:
                image, boxes, labels = self._random_expand_crop(image, boxes, labels)
                cur_w, cur_h = image.size  # update current working canvas dimensions

            # 3. Random Horizontal Flip (50% probability - spatial geometry)
            if random.random() > 0.5 and len(boxes) > 0:
                image = F.hflip(image)
                xmin = boxes[:, 0].clone()
                xmax = boxes[:, 2].clone()
                boxes[:, 0] = cur_w - xmax
                boxes[:, 2] = cur_w - xmin

        # 4. Resize (Multi-scale for training, fixed for val)
        if self.multi_scale:
            size = random.choice(MULTI_SCALE_SIZES)
            target_h, target_w = size, size
        else:
            target_h, target_w = self.target_size

        # Resize image to target size
        image = F.resize(image, [target_h, target_w])

        # 5. Scale & Filter Box
        scale_x = target_w / float(cur_w)
        scale_y = target_h / float(cur_h)

        if len(boxes) > 0:
            boxes[:, 0] = boxes[:, 0] * scale_x
            boxes[:, 1] = boxes[:, 1] * scale_y
            boxes[:, 2] = boxes[:, 2] * scale_x
            boxes[:, 3] = boxes[:, 3] * scale_y

            # Clamp boxes to image boundaries
            boxes[:, 0] = boxes[:, 0].clamp(min=0, max=target_w)
            boxes[:, 1] = boxes[:, 1].clamp(min=0, max=target_h)
            boxes[:, 2] = boxes[:, 2].clamp(min=0, max=target_w)
            boxes[:, 3] = boxes[:, 3].clamp(min=0, max=target_h)

            # Filter valid boxes (min 2px each side after resize)
            valid_mask = (boxes[:, 2] > boxes[:, 0] + 2) & (boxes[:, 3] > boxes[:, 1] + 2)
            boxes = boxes[valid_mask]
            labels = labels[valid_mask]

        # 6. Normalize (ToTensor and ImageNet normalization)
        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, mean=self.mean, std=self.std)

        return image_tensor, boxes, labels, (raw_orig_h, raw_orig_w)

    def _random_expand_crop(self, image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor):
        """
        SSD-style random expand then random crop that guarantees at least one
        GT box center is kept inside the crop, keeping boxes and labels synchronized.
        """
        width, height = image.size

        # Random expand: place image on a larger canvas (1x to 2x)
        expand_ratio = random.uniform(1.0, 2.0)
        new_w = int(width * expand_ratio)
        new_h = int(height * expand_ratio)

        # Create canvas with mean pixel values
        mean_pixel = tuple(int(m * 255) for m in self.mean)
        canvas = Image.new("RGB", (new_w, new_h), mean_pixel)

        # Random offset to place original image
        left = random.randint(0, new_w - width)
        top = random.randint(0, new_h - height)
        canvas.paste(image, (left, top))

        # Shift boxes accordingly
        shifted_boxes = boxes.clone()
        shifted_boxes[:, 0] += left
        shifted_boxes[:, 1] += top
        shifted_boxes[:, 2] += left
        shifted_boxes[:, 3] += top

        expanded_image = canvas
        exp_w, exp_h = new_w, new_h

        # Random crop: ensure at least one box center is inside crop
        for _ in range(50):  # max attempts
            crop_w = random.randint(int(0.5 * exp_w), exp_w)
            crop_h = random.randint(int(0.5 * exp_h), exp_h)
            crop_x = random.randint(0, exp_w - crop_w)
            crop_y = random.randint(0, exp_h - crop_h)

            # Check if at least one box center is inside crop
            centers_x = (shifted_boxes[:, 0] + shifted_boxes[:, 2]) / 2.0
            centers_y = (shifted_boxes[:, 1] + shifted_boxes[:, 3]) / 2.0
            inside = (
                (centers_x >= crop_x)
                & (centers_x <= crop_x + crop_w)
                & (centers_y >= crop_y)
                & (centers_y <= crop_y + crop_h)
            )

            if inside.any():
                # Adjust boxes to crop coordinates and filter
                cropped_boxes = shifted_boxes.clone()
                cropped_boxes[:, 0] = (cropped_boxes[:, 0] - crop_x).clamp(min=0, max=crop_w)
                cropped_boxes[:, 1] = (cropped_boxes[:, 1] - crop_y).clamp(min=0, max=crop_h)
                cropped_boxes[:, 2] = (cropped_boxes[:, 2] - crop_x).clamp(min=0, max=crop_w)
                cropped_boxes[:, 3] = (cropped_boxes[:, 3] - crop_y).clamp(min=0, max=crop_h)

                # Keep only boxes that are still valid and whose center was inside
                valid = inside & (cropped_boxes[:, 2] > cropped_boxes[:, 0] + 2) & (cropped_boxes[:, 3] > cropped_boxes[:, 1] + 2)
                if valid.any():
                    # Only crop image AFTER confirming valid boxes exist
                    cropped_image = expanded_image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                    return cropped_image, cropped_boxes[valid], labels[valid]

        # Fallback: return expanded image with shifted boxes (no crop applied)
        return expanded_image, shifted_boxes, labels

