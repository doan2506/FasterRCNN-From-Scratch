import torch
from utils.box_utils import box_area

try:
    from torchvision.ops import nms as tv_nms
    from torchvision.ops import batched_nms as tv_batched_nms
except ImportError:
    tv_nms = None
    tv_batched_nms = None


def pure_pytorch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Pure PyTorch Non-Maximum Suppression (NMS) algorithm.
    boxes: Tensor (N, 4) in [xmin, ymin, xmax, ymax]
    scores: Tensor (N,) confidence scores
    iou_threshold: Float threshold for IoU suppression
    Returns: Tensor of kept indices
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break

        i = order[0].item()
        keep.append(i)

        # Compute IoU of box i with remaining boxes in order[1:]
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        rem_areas = areas[order[1:]]
        union = areas[i] + rem_areas - inter
        union = torch.clamp(union, min=1e-6)

        iou = inter / union
        mask = iou <= iou_threshold

        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Non-Maximum Suppression (NMS) with C++/CUDA acceleration when torchvision is available,
    falling back to pure PyTorch implementation.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    if tv_nms is not None:
        return tv_nms(boxes, scores, iou_threshold)
    return pure_pytorch_nms(boxes, scores, iou_threshold)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float
) -> torch.Tensor:
    """
    Per-class Non-Maximum Suppression.
    boxes: Tensor (N, 4)
    scores: Tensor (N,)
    labels: Tensor (N,)
    iou_threshold: Float
    Returns: Tensor of kept indices
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    if tv_batched_nms is not None:
        return tv_batched_nms(boxes, scores, labels, iou_threshold)

    # Pure PyTorch fallback: Offset boxes per class
    max_coordinate = boxes.max()
    offsets = labels.to(boxes.dtype) * (max_coordinate + 1.0)
    boxes_for_nms = boxes + offsets[:, None]

    return pure_pytorch_nms(boxes_for_nms, scores, iou_threshold)


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.01,
    method: str = "gaussian",
) -> tuple:
    """
    Soft Non-Maximum Suppression (Soft-NMS).
    Decays detection scores of overlapping boxes instead of instantly discarding them.

    boxes: Tensor (N, 4) in [xmin, ymin, xmax, ymax]
    scores: Tensor (N,) confidence scores
    iou_threshold: IoU threshold for linear decay (default: 0.5)
    sigma: Gaussian bandwidth parameter (default: 0.5)
    score_threshold: Pruning score threshold to discard low-confidence boxes (default: 0.01)
    method: 'gaussian', 'linear', or 'hard'

    Returns:
        keep_indices: Tensor of kept box original indices
        updated_scores: Tensor of updated confidence scores for kept boxes
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device), torch.empty((0,), dtype=scores.dtype, device=boxes.device)

    boxes = boxes.clone()
    scores = scores.clone()
    num_boxes = boxes.shape[0]
    indices = torch.arange(num_boxes, dtype=torch.int64, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    keep_indices = []
    keep_scores = []

    for _ in range(num_boxes):
        if scores.numel() == 0:
            break

        max_pos = torch.argmax(scores)
        max_score = scores[max_pos]

        if max_score < score_threshold:
            break

        # Keep current maximum box
        keep_indices.append(indices[max_pos])
        keep_scores.append(max_score)

        if scores.numel() == 1:
            break

        # Extract best box coordinates and area
        b_x1 = x1[max_pos]
        b_y1 = y1[max_pos]
        b_x2 = x2[max_pos]
        b_y2 = y2[max_pos]
        b_area = areas[max_pos]

        # Remaining boxes mask
        mask = torch.ones(len(scores), dtype=torch.bool, device=boxes.device)
        mask[max_pos] = False

        x1 = x1[mask]
        y1 = y1[mask]
        x2 = x2[mask]
        y2 = y2[mask]
        areas = areas[mask]
        scores = scores[mask]
        indices = indices[mask]

        # Compute IoU between best box and remaining boxes
        xx1 = torch.maximum(b_x1, x1)
        yy1 = torch.maximum(b_y1, y1)
        xx2 = torch.minimum(b_x2, x2)
        yy2 = torch.minimum(b_y2, y2)

        w = (xx2 - xx1).clamp(min=0.0)
        h = (yy2 - yy1).clamp(min=0.0)
        inter = w * h

        union = b_area + areas - inter
        union = torch.clamp(union, min=1e-6)
        ious = inter / union

        # Score decay
        if method == "gaussian":
            decay = torch.exp(-(ious ** 2) / sigma)
            scores = scores * decay
        elif method == "linear":
            decay = torch.where(ious >= iou_threshold, 1.0 - ious, torch.ones_like(ious))
            scores = scores * decay
        else:  # hard nms
            keep_mask = ious < iou_threshold
            x1 = x1[keep_mask]
            y1 = y1[keep_mask]
            x2 = x2[keep_mask]
            y2 = y2[keep_mask]
            areas = areas[keep_mask]
            scores = scores[keep_mask]
            indices = indices[keep_mask]
            continue

        # Prune low-scoring boxes
        valid_mask = scores >= score_threshold
        x1 = x1[valid_mask]
        y1 = y1[valid_mask]
        x2 = x2[valid_mask]
        y2 = y2[valid_mask]
        areas = areas[valid_mask]
        scores = scores[valid_mask]
        indices = indices[valid_mask]

    if len(keep_indices) == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device), torch.empty((0,), dtype=scores.dtype, device=boxes.device)

    return torch.stack(keep_indices), torch.stack(keep_scores)


def batched_soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.01,
    method: str = "gaussian",
) -> tuple:
    """
    Per-class Soft-NMS.
    boxes: Tensor (N, 4)
    scores: Tensor (N,)
    labels: Tensor (N,)
    Returns:
        kept_boxes: Tensor (K, 4)
        kept_scores: Tensor (K,)
        kept_labels: Tensor (K,)
    """
    if boxes.numel() == 0:
        return (
            torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device),
            torch.empty((0,), dtype=scores.dtype, device=scores.device),
            torch.empty((0,), dtype=labels.dtype, device=labels.device),
        )

    unique_labels = torch.unique(labels)
    all_boxes = []
    all_scores = []
    all_labels = []

    for lbl in unique_labels:
        cls_mask = labels == lbl
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        cls_keep_idx, cls_updated_scores = soft_nms(
            cls_boxes,
            cls_scores,
            iou_threshold=iou_threshold,
            sigma=sigma,
            score_threshold=score_threshold,
            method=method,
        )

        if len(cls_keep_idx) > 0:
            all_boxes.append(cls_boxes[cls_keep_idx])
            all_scores.append(cls_updated_scores)
            all_labels.append(torch.full((len(cls_keep_idx),), lbl.item(), dtype=labels.dtype, device=labels.device))

    if len(all_boxes) == 0:
        return (
            torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device),
            torch.empty((0,), dtype=scores.dtype, device=scores.device),
            torch.empty((0,), dtype=labels.dtype, device=labels.device),
        )

    cat_boxes = torch.cat(all_boxes, dim=0)
    cat_scores = torch.cat(all_scores, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)

    # Sort descending by updated score
    sort_idx = torch.argsort(cat_scores, descending=True)
    return cat_boxes[sort_idx], cat_scores[sort_idx], cat_labels[sort_idx]

