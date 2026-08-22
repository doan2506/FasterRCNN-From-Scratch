import os
import glob
import json
import argparse
import urllib.request
import torch
import torchvision.transforms.functional as F
from PIL import Image
from models.faster_rcnn import FasterRCNN
from utils.dataset import CLASSES, IDX_TO_CLASS
from utils.nms import batched_nms, batched_soft_nms


# Default direct download URL for best.pth (can be customized with release URL)
DEFAULT_WEIGHTS_URL = "https://github.com/doan2506/FasterRCNN-From-Scratch/releases/download/v1.0/best.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN Inference Script from Scratch")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images for inference")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output JSON path")
    parser.add_argument("--model_path", type=str, default="./models/best.pth", help="Path to model checkpoint")
    parser.add_argument("--weights_url", type=str, default=DEFAULT_WEIGHTS_URL, help="URL to download weights if model_path is missing")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50"], help="Backbone architecture")
    parser.add_argument("--conf_thresh", type=float, default=0.05, help="Confidence score threshold")
    parser.add_argument("--nms_thresh", type=float, default=0.5, help="NMS IoU threshold (default: 0.5)")
    parser.add_argument("--img_size", type=int, default=640, help="Inference image resize target")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for faster inference")
    parser.add_argument("--use_tta", action="store_true", default=True, help="Enable Test-Time Augmentation (default: True)")
    parser.add_argument("--no_tta", dest="use_tta", action="store_false", help="Disable Test-Time Augmentation")
    parser.add_argument("--use_soft_nms", action="store_true", default=False, help="Enable Soft-NMS (default: False)")
    parser.add_argument("--soft_nms_sigma", type=float, default=0.5, help="Soft-NMS Gaussian sigma (default: 0.5)")
    parser.add_argument("--soft_nms_method", type=str, default="gaussian", choices=["gaussian", "linear", "hard"], help="Soft-NMS method (default: gaussian)")
    return parser.parse_args()


def download_weights(url: str, dest_path: str):
    """
    Auto-downloads weights if not found locally, as required by the submission guidelines.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    print(f"[INFO] Checkpoint not found at {dest_path}. Automatically downloading from:\n   {url} ...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"[OK] Successfully downloaded weights to {dest_path}")
    except Exception as e:
        print(f"[WARNING] Failed to auto-download weights from {url}: {e}")
        print("   Please make sure the checkpoint is available or provide a valid --weights_url.")


def load_model(
    model_path: str,
    backbone: str,
    conf_thresh: float,
    nms_thresh: float,
    device: torch.device,
    weights_url: str = DEFAULT_WEIGHTS_URL,
    use_soft_nms: bool = False,
    soft_nms_sigma: float = 0.5,
    soft_nms_method: str = "gaussian",
):
    # Auto-download weights if missing
    if not os.path.exists(model_path):
        download_weights(weights_url, model_path)

    checkpoint = None
    if os.path.exists(model_path):
        print(f"Loading checkpoint weights from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except Exception:
            checkpoint = torch.load(model_path, map_location=device)

    detected_backbone = backbone
    detected_fc_dim = 1024
    detected_ratios = (0.5, 1.0, 2.0)
    detected_scales = (1.0,)
    state_dict = None

    if checkpoint is not None:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            if "args" in checkpoint and isinstance(checkpoint["args"], dict):
                detected_backbone = checkpoint["args"].get("backbone", detected_backbone)
        else:
            state_dict = checkpoint

    if state_dict is not None:
        if "roi_heads.box_head.fc6.weight" in state_dict:
            detected_fc_dim = state_dict["roi_heads.box_head.fc6.weight"].shape[0]
        if "rpn.anchor_generator.ratios" in state_dict:
            detected_ratios = tuple(state_dict["rpn.anchor_generator.ratios"].cpu().tolist())
        if "rpn.anchor_generator.scales" in state_dict:
            detected_scales = tuple(state_dict["rpn.anchor_generator.scales"].cpu().tolist())
        elif "rpn.head.cls_logits.weight" in state_dict:
            num_anchors = state_dict["rpn.head.cls_logits.weight"].shape[0]
            if num_anchors == len(detected_ratios):
                detected_scales = (1.0,)

    print(f"Instantiating model: FASTER R-CNN (Backbone: {detected_backbone}, FC Dim: {detected_fc_dim}, Anchors/Loc: {len(detected_ratios)*len(detected_scales)})")

    model = FasterRCNN(
        num_classes=len(CLASSES),
        backbone_name=detected_backbone,
        ratios=detected_ratios,
        scales=detected_scales,
        fc_dim=detected_fc_dim,
        dropout_p=0.0,
        pretrained=False,
        conf_threshold=conf_thresh,
        nms_threshold=nms_thresh,
        use_soft_nms=use_soft_nms,
        soft_nms_sigma=soft_nms_sigma,
        soft_nms_method=soft_nms_method,
    )

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
        print("[OK] Checkpoint weights successfully loaded.")
    else:
        print(f"[WARNING] Checkpoint {model_path} not found! Initializing with random/pretrained weights.")

    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str, target_size: int):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    # Resize image
    resized_img = F.resize(image, [target_size, target_size])
    image_tensor = F.to_tensor(resized_img)
    image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return image_tensor, orig_w, orig_h


@torch.no_grad()
def infer_batch(model, batch_tensors, device, use_tta=False):
    """
    Runs model inference on a batch of image tensors, optionally applying TTA (Horizontal Flip).
    """
    batch_tensors = batch_tensors.to(device)
    results = model(batch_tensors)

    if not use_tta:
        return results

    # Test Time Augmentation: Flip horizontally
    flipped_tensors = torch.flip(batch_tensors, dims=[-1])
    flipped_results = model(flipped_tensors)

    img_w = batch_tensors.shape[-1]
    merged_results = []
    for orig_res, flip_res in zip(results, flipped_results):
        o_boxes = orig_res["boxes"]
        o_scores = orig_res["scores"]
        o_labels = orig_res["labels"]

        f_boxes = flip_res["boxes"]
        f_scores = flip_res["scores"]
        f_labels = flip_res["labels"]

        if len(f_boxes) > 0:
            # Unflip horizontal coordinates
            x1 = img_w - f_boxes[:, 2]
            x2 = img_w - f_boxes[:, 0]
            f_boxes_unflipped = torch.stack([x1, f_boxes[:, 1], x2, f_boxes[:, 3]], dim=1)

            all_boxes = torch.cat([o_boxes, f_boxes_unflipped], dim=0)
            all_scores = torch.cat([o_scores, f_scores], dim=0)
            all_labels = torch.cat([o_labels, f_labels], dim=0)
        else:
            all_boxes, all_scores, all_labels = o_boxes, o_scores, o_labels

        if len(all_boxes) > 0:
            if hasattr(model.roi_heads, "use_soft_nms") and model.roi_heads.use_soft_nms:
                m_boxes, m_scores, m_labels = batched_soft_nms(
                    all_boxes,
                    all_scores,
                    all_labels,
                    iou_threshold=model.nms_threshold,
                    sigma=model.roi_heads.soft_nms_sigma,
                    score_threshold=model.conf_threshold,
                    method=model.roi_heads.soft_nms_method,
                )
            else:
                keep = batched_nms(all_boxes, all_scores, all_labels, model.nms_threshold)
                m_boxes, m_scores, m_labels = all_boxes[keep], all_scores[keep], all_labels[keep]

            if len(m_boxes) > model.max_detections_per_img:
                m_boxes = m_boxes[: model.max_detections_per_img]
                m_scores = m_scores[: model.max_detections_per_img]
                m_labels = m_labels[: model.max_detections_per_img]

            merged_results.append({
                "boxes": m_boxes,
                "scores": m_scores,
                "labels": m_labels,
            })
        else:
            merged_results.append(orig_res)

    return merged_results


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        args.model_path,
        args.backbone,
        args.conf_thresh,
        args.nms_thresh,
        device,
        args.weights_url,
        use_soft_nms=args.use_soft_nms,
        soft_nms_sigma=args.soft_nms_sigma,
        soft_nms_method=args.soft_nms_method,
    )

    # Gather all image files
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))

    # Sort paths for consistent order
    image_paths = sorted(list(set(image_paths)))
    print(f"Found {len(image_paths)} images in {args.image_dir}.")

    predictions = []

    # Process in batches for fast inference
    for i in range(0, len(image_paths), args.batch_size):
        chunk_paths = image_paths[i : i + args.batch_size]
        batch_tensors = []
        meta_info = []

        for p in chunk_paths:
            img_tensor, orig_w, orig_h = preprocess_image(p, args.img_size)
            batch_tensors.append(img_tensor)
            meta_info.append((os.path.basename(p), orig_w, orig_h))

        batch_stack = torch.stack(batch_tensors, dim=0)
        batch_results = infer_batch(model, batch_stack, device, use_tta=args.use_tta)

        for det_results, (img_id, orig_w, orig_h) in zip(batch_results, meta_info):
            boxes = det_results["boxes"].cpu()
            scores = det_results["scores"].cpu()
            labels = det_results["labels"].cpu()

            boxes_list = []
            if len(boxes) > 0:
                scale_x = orig_w / float(args.img_size)
                scale_y = orig_h / float(args.img_size)

                boxes[:, 0] = (boxes[:, 0] * scale_x).clamp(min=0, max=orig_w)
                boxes[:, 1] = (boxes[:, 1] * scale_y).clamp(min=0, max=orig_h)
                boxes[:, 2] = (boxes[:, 2] * scale_x).clamp(min=0, max=orig_w)
                boxes[:, 3] = (boxes[:, 3] * scale_y).clamp(min=0, max=orig_h)

                for box, score, label_idx in zip(boxes, scores, labels):
                    label_num = label_idx.item()
                    cls_name = IDX_TO_CLASS[label_num] if label_num in IDX_TO_CLASS else CLASSES[label_num]
                    xmin, ymin, xmax, ymax = box.tolist()

                    # Filter out degenerate/empty boxes
                    if xmax <= xmin + 0.1 or ymax <= ymin + 0.1:
                        continue

                    boxes_list.append({
                        "class": cls_name,
                        "confidence": round(float(score.item()), 4),
                        "bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
                    })

            predictions.append({
                "image_id": img_id,
                "boxes": boxes_list,
            })

    # Save to JSON output
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"Successfully exported {len(predictions)} image predictions to {args.output}")


if __name__ == "__main__":
    main()
