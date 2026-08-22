import math
import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of a set of bounding boxes.
    boxes: Tensor of shape (N, 4) in format [xmin, ymin, xmax, ymax]
    """
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise Intersection over Union (IoU) between two sets of boxes.
    boxes1: Tensor (N, 4) [xmin, ymin, xmax, ymax]
    boxes2: Tensor (M, 4) [xmin, ymin, xmax, ymax]
    Returns: Tensor (N, M) of IoU values.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    union = torch.clamp(union, min=1e-6)

    return inter / union


def box_transform(anchors: torch.Tensor, gt_boxes: torch.Tensor, weights=(1.0, 1.0, 1.0, 1.0)) -> torch.Tensor:
    """
    Encodes bounding boxes relative to anchors.
    anchors: Tensor (N, 4) [xmin, ymin, xmax, ymax]
    gt_boxes: Tensor (N, 4) [xmin, ymin, xmax, ymax]
    Returns: deltas (N, 4) [dx, dy, dw, dh]
    """
    wx, wy, ww, wh = weights

    anchor_w = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-5)
    anchor_h = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-5)
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_w
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_h

    gt_w = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-5)
    gt_h = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-5)
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_w
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_h

    dx = wx * (gt_ctr_x - anchor_ctr_x) / anchor_w
    dy = wy * (gt_ctr_y - anchor_ctr_y) / anchor_h
    dw = ww * torch.log((gt_w / anchor_w).clamp(min=1e-6))
    dh = wh * torch.log((gt_h / anchor_h).clamp(min=1e-6))

    return torch.stack([dx, dy, dw, dh], dim=1)


def box_decode(anchors: torch.Tensor, deltas: torch.Tensor, weights=(1.0, 1.0, 1.0, 1.0)) -> torch.Tensor:
    """
    Decodes regression deltas applied to anchors back into absolute coordinates.
    anchors: Tensor (N, 4) or (B, N, 4)
    deltas: Tensor (N, 4) or (B, N, 4)
    Returns: boxes (N, 4) or (B, N, 4) in [xmin, ymin, xmax, ymax]
    """
    wx, wy, ww, wh = weights

    anchor_w = (anchors[..., 2] - anchors[..., 0]).clamp(min=1e-5)
    anchor_h = (anchors[..., 3] - anchors[..., 1]).clamp(min=1e-5)
    anchor_ctr_x = anchors[..., 0] + 0.5 * anchor_w
    anchor_ctr_y = anchors[..., 1] + 0.5 * anchor_h

    dx = deltas[..., 0] / wx
    dy = deltas[..., 1] / wy
    dw = deltas[..., 2] / ww
    dh = deltas[..., 3] / wh

    # Prevent overflow in exp
    dw = torch.clamp(dw, min=-10.0, max=10.0)
    dh = torch.clamp(dh, min=-10.0, max=10.0)

    pred_ctr_x = dx * anchor_w + anchor_ctr_x
    pred_ctr_y = dy * anchor_h + anchor_ctr_y
    pred_w = torch.exp(dw) * anchor_w
    pred_h = torch.exp(dh) * anchor_h

    xmin = pred_ctr_x - 0.5 * pred_w
    ymin = pred_ctr_y - 0.5 * pred_h
    xmax = pred_ctr_x + 0.5 * pred_w
    ymax = pred_ctr_y + 0.5 * pred_h

    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def clip_boxes(boxes: torch.Tensor, image_shape: tuple) -> torch.Tensor:
    """
    Clips bounding boxes to image boundaries.
    boxes: Tensor (N, 4) [xmin, ymin, xmax, ymax]
    image_shape: (height, width)
    """
    h, w = image_shape[:2]
    boxes[..., 0] = boxes[..., 0].clamp(min=0, max=w)
    boxes[..., 1] = boxes[..., 1].clamp(min=0, max=h)
    boxes[..., 2] = boxes[..., 2].clamp(min=0, max=w)
    boxes[..., 3] = boxes[..., 3].clamp(min=0, max=h)
    return boxes


def bbox_iou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    loss_type: str = "ciou",
    reduction: str = "mean",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes IoU-based bounding box regression loss (GIoU, DIoU, CIoU, standard IoU)
    between predicted boxes and target ground-truth boxes.

    pred_boxes: Tensor (N, 4) in format [xmin, ymin, xmax, ymax]
    target_boxes: Tensor (N, 4) in format [xmin, ymin, xmax, ymax]
    loss_type: 'ciou', 'giou', 'diou', or 'iou'
    reduction: 'mean', 'sum', or 'none'
    Returns: Loss tensor
    """
    if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
        if reduction == "none":
            return torch.empty((0,), device=pred_boxes.device)
        return torch.tensor(0.0, device=pred_boxes.device)

    # 1. Coordinate unpacking
    p_x1, p_y1, p_x2, p_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    t_x1, t_y1, t_x2, t_y2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

    p_w = (p_x2 - p_x1).clamp(min=eps)
    p_h = (p_y2 - p_y1).clamp(min=eps)
    t_w = (t_x2 - t_x1).clamp(min=eps)
    t_h = (t_y2 - t_y1).clamp(min=eps)

    p_area = p_w * p_h
    t_area = t_w * t_h

    # 2. Intersection Area
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    # 3. Union Area & IoU
    union_area = p_area + t_area - inter_area
    iou = inter_area / union_area.clamp(min=eps)

    loss_type = loss_type.lower()
    if loss_type == "iou":
        loss = 1.0 - iou
    elif loss_type == "giou":
        # Smallest enclosing box
        c_x1 = torch.min(p_x1, t_x1)
        c_y1 = torch.min(p_y1, t_y1)
        c_x2 = torch.max(p_x2, t_x2)
        c_y2 = torch.max(p_y2, t_y2)
        c_area = (c_x2 - c_x1).clamp(min=0.0) * (c_y2 - c_y1).clamp(min=0.0)
        giou = iou - (c_area - union_area) / c_area.clamp(min=eps)
        loss = 1.0 - giou
    elif loss_type in ["diou", "ciou"]:
        # Smallest enclosing box diagonal
        c_x1 = torch.min(p_x1, t_x1)
        c_y1 = torch.min(p_y1, t_y1)
        c_x2 = torch.max(p_x2, t_x2)
        c_y2 = torch.max(p_y2, t_y2)
        c_diag_sq = (c_x2 - c_x1).clamp(min=0.0) ** 2 + (c_y2 - c_y1).clamp(min=0.0) ** 2 + eps

        # Center distance squared
        p_ctr_x = (p_x1 + p_x2) * 0.5
        p_ctr_y = (p_y1 + p_y2) * 0.5
        t_ctr_x = (t_x1 + t_x2) * 0.5
        t_ctr_y = (t_y1 + t_y2) * 0.5
        rho_sq = (p_ctr_x - t_ctr_x) ** 2 + (p_ctr_y - t_ctr_y) ** 2

        diou = iou - rho_sq / c_diag_sq
        if loss_type == "diou":
            loss = 1.0 - diou
        else:  # CIoU
            v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(t_w / t_h) - torch.atan(p_w / p_h), 2)
            with torch.no_grad():
                alpha = v / (1.0 - iou + v + eps)
            ciou = diou - alpha * v
            loss = 1.0 - ciou
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose from 'ciou', 'giou', 'diou', 'iou'.")

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return bbox_iou_loss(pred_boxes, target_boxes, loss_type="giou", reduction=reduction)


def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return bbox_iou_loss(pred_boxes, target_boxes, loss_type="ciou", reduction=reduction)


def diou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return bbox_iou_loss(pred_boxes, target_boxes, loss_type="diou", reduction=reduction)

