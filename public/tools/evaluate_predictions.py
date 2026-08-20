#!/usr/bin/env python3
"""Validate object-detection predictions and compute VOC-style mAP@0.5."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground_truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_detections_per_image", type=int, default=100)
    parser.add_argument("--allow_missing_images", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_bbox(bbox: Any, image: dict[str, Any], context: str) -> list[float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox in {context}: expected [xmin, ymin, xmax, ymax].")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox):
        raise ValueError(f"Invalid bbox in {context}: coordinates must be numeric.")
    xmin, ymin, xmax, ymax = map(float, bbox)
    if not (0 <= xmin < xmax <= image["width"] and 0 <= ymin < ymax <= image["height"]):
        raise ValueError(f"Invalid bbox in {context}: coordinates outside image bounds or empty box.")
    return [xmin, ymin, xmax, ymax]


def validate_ground_truth(data: Any) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if not isinstance(data, dict):
        raise ValueError("Ground truth must be a JSON object.")
    classes, images, annotations = (data.get(name) for name in ("classes", "images", "annotations"))
    if not isinstance(classes, list) or not classes or not all(isinstance(item, str) for item in classes):
        raise ValueError("Ground truth field 'classes' must be a non-empty list of strings.")
    if len(classes) != len(set(classes)):
        raise ValueError("Ground truth classes must be unique.")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("Ground truth fields 'images' and 'annotations' must be lists.")
    image_info: dict[str, dict[str, Any]] = {}
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("Each ground truth image entry must be an object.")
        image_id, width, height = image.get("id"), image.get("width"), image.get("height")
        if not isinstance(image_id, str) or not image_id or image_id in image_info:
            raise ValueError(f"Invalid or duplicate image id: {image_id}")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError(f"Image {image_id} has invalid width.")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError(f"Image {image_id} has invalid height.")
        image_info[image_id] = image
    class_set = set(classes)
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise ValueError(f"Ground truth annotation {index} must be an object.")
        image_id, class_name = annotation.get("image_id"), annotation.get("class")
        if image_id not in image_info:
            raise ValueError(f"Annotation references unknown image_id: {image_id}")
        if class_name not in class_set:
            raise ValueError(f"Annotation uses unknown class: {class_name}")
        validate_bbox(annotation.get("bbox"), image_info[image_id], f"ground truth {image_id}")
    return classes, image_info


def normalize_predictions(
    data: Any,
    classes: list[str],
    image_info: dict[str, dict[str, Any]],
    max_detections: int,
    require_complete: bool,
) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("Predictions must be a JSON array.")
    class_set, seen, normalized = set(classes), set(), []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("Each prediction entry must be an object.")
        image_id, boxes = entry.get("image_id"), entry.get("boxes")
        if image_id not in image_info:
            raise ValueError(f"Prediction references unknown image_id: {image_id}")
        if image_id in seen:
            raise ValueError(f"Duplicate prediction entry for image_id: {image_id}")
        if not isinstance(boxes, list):
            raise ValueError(f"Prediction for {image_id} must contain a boxes list.")
        seen.add(image_id)
        image_boxes = []
        for index, box in enumerate(boxes):
            if not isinstance(box, dict):
                raise ValueError(f"Prediction box {index} for {image_id} must be an object.")
            class_name, confidence = box.get("class"), box.get("confidence")
            if class_name not in class_set:
                raise ValueError(f"Prediction for {image_id} uses unknown class: {class_name}")
            if (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not 0 <= confidence <= 1
            ):
                raise ValueError(f"Prediction for {image_id} has invalid confidence: {confidence}")
            image_boxes.append(
                {
                    "image_id": image_id,
                    "class": class_name,
                    "confidence": float(confidence),
                    "bbox": validate_bbox(box.get("bbox"), image_info[image_id], f"prediction {image_id}"),
                }
            )
        image_boxes.sort(key=lambda item: item["confidence"], reverse=True)
        normalized.extend(image_boxes[:max_detections])
    if require_complete:
        missing = sorted(set(image_info) - seen)
        if missing:
            raise ValueError(f"Predictions are missing {len(missing)} image(s): {', '.join(missing[:10])}")
    return normalized


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec, mpre = [0.0] + recalls + [1.0], [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    return sum(
        (mrec[index] - mrec[index - 1]) * mpre[index]
        for index in range(1, len(mrec))
        if mrec[index] != mrec[index - 1]
    )


def performance_score(map_50: float) -> int:
    """Match the published 0.30/0.45/0.60/0.75 rubric exactly."""
    if map_50 < 0.30:
        return 0
    if map_50 < 0.45:
        return 5
    if map_50 < 0.60:
        return 10
    if map_50 < 0.75:
        return 15
    return 20


def evaluate(
    ground_truth: dict[str, Any], predictions: list[dict[str, Any]], classes: list[str], iou_threshold: float
) -> dict[str, Any]:
    gt_by_class: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: defaultdict(list) for name in classes
    }
    for annotation in ground_truth["annotations"]:
        gt_by_class[annotation["class"]][annotation["image_id"]].append(
            {"bbox": list(map(float, annotation["bbox"])), "matched": False}
        )
    pred_by_class = {name: [] for name in classes}
    for prediction in predictions:
        pred_by_class[prediction["class"]].append(prediction)

    per_class, aps = {}, []
    total_tp = total_fp = total_gt = 0
    for class_name in classes:
        class_gt = gt_by_class[class_name]
        num_gt = sum(map(len, class_gt.values()))
        class_predictions = sorted(pred_by_class[class_name], key=lambda item: item["confidence"], reverse=True)
        tp_flags, fp_flags = [], []
        for prediction in class_predictions:
            candidates = class_gt.get(prediction["image_id"], [])
            best_iou, best_index = 0.0, -1
            for index, target in enumerate(candidates):
                if target["matched"]:
                    continue
                iou = bbox_iou(prediction["bbox"], target["bbox"])
                if iou > best_iou:
                    best_iou, best_index = iou, index
            matched = best_index >= 0 and best_iou >= iou_threshold
            if matched:
                candidates[best_index]["matched"] = True
            tp_flags.append(int(matched))
            fp_flags.append(int(not matched))
        cumulative_tp, cumulative_fp, tp_sum, fp_sum = [], [], 0, 0
        for tp, fp in zip(tp_flags, fp_flags):
            tp_sum, fp_sum = tp_sum + tp, fp_sum + fp
            cumulative_tp.append(tp_sum)
            cumulative_fp.append(fp_sum)
        recalls = [value / num_gt if num_gt else 0.0 for value in cumulative_tp]
        precisions = [tp / max(tp + fp, 1) for tp, fp in zip(cumulative_tp, cumulative_fp)]
        ap = compute_ap(recalls, precisions) if num_gt else 0.0
        if num_gt:
            aps.append(ap)
        total_tp, total_fp, total_gt = total_tp + tp_sum, total_fp + fp_sum, total_gt + num_gt
        per_class[class_name] = {
            "ap": round(ap, 6),
            "num_ground_truth": num_gt,
            "num_predictions": len(class_predictions),
            "true_positives": tp_sum,
            "false_positives": fp_sum,
            "recall": round(tp_sum / num_gt, 6) if num_gt else 0.0,
            "precision": round(tp_sum / max(tp_sum + fp_sum, 1), 6),
        }
    map_50 = sum(aps) / len(aps) if aps else 0.0
    return {
        "mAP@0.5": round(map_50, 6),
        "performance_points": performance_score(map_50),
        "iou_threshold": iou_threshold,
        "num_ground_truth_boxes": total_gt,
        "num_predictions": len(predictions),
        "micro_precision": round(total_tp / max(total_tp + total_fp, 1), 6),
        "micro_recall": round(total_tp / total_gt, 6) if total_gt else 0.0,
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    ground_truth = load_json(args.ground_truth)
    classes, image_info = validate_ground_truth(ground_truth)
    predictions = normalize_predictions(
        load_json(args.predictions), classes, image_info, args.max_detections_per_image, not args.allow_missing_images
    )
    result = evaluate(ground_truth, predictions, classes, args.iou_threshold)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
