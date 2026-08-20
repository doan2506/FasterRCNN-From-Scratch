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
