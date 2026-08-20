import torch
import torch.nn as nn
import torch.nn.functional as F
from models.roi_align import MultiScaleRoIAlign
from utils.box_utils import box_iou, box_transform, box_decode, clip_boxes
from utils.nms import batched_nms


class FastRCNNHead(nn.Module):
    """
    Standard Two-FC MLP Head for Fast R-CNN classification and class-specific bounding box regression.
    """

    def __init__(self, in_channels=256, roi_size=(7, 7), num_classes=5, fc_dim=1024):
        super().__init__()
        self.num_classes = num_classes
        in_dim = in_channels * roi_size[0] * roi_size[1]

        self.fc6 = nn.Linear(in_dim, fc_dim)
        self.fc7 = nn.Linear(fc_dim, fc_dim)

        # num_classes + 1 (background is last class index: num_classes)
        self.cls_score = nn.Linear(fc_dim, num_classes + 1)
        self.bbox_pred = nn.Linear(fc_dim, (num_classes + 1) * 4)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, roi_features: torch.Tensor) -> tuple:
        """
        roi_features: (N_rois, C, 7, 7)
        Returns:
            cls_scores: (N_rois, num_classes + 1)
            bbox_deltas: (N_rois, (num_classes + 1) * 4)
        """
        x = roi_features.flatten(start_dim=1)
        x = F.relu(self.fc6(x), inplace=False)
        x = F.relu(self.fc7(x), inplace=False)

        cls_scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return cls_scores, bbox_deltas


class RoIHeads(nn.Module):
    """
    Faster R-CNN RoI Head combining Multi-Scale RoIAlign, 2-FC Head,
    RoI Target Sampling, Loss Computation, and Inference Post-processing.
    """

    def __init__(
        self,
        in_channels=256,
        num_classes=5,
        roi_size=(7, 7),
        fc_dim=1024,
        batch_size_per_image=512,
        positive_fraction=0.25,
        fg_iou_thresh=0.5,
        bg_iou_thresh_hi=0.5,
        bg_iou_thresh_lo=0.0,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
        bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bg_class_idx = num_classes
        self.roi_align = MultiScaleRoIAlign(output_size=roi_size)
        self.box_head = FastRCNNHead(in_channels=in_channels, roi_size=roi_size, num_classes=num_classes, fc_dim=fc_dim)

        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh_hi = bg_iou_thresh_hi
        self.bg_iou_thresh_lo = bg_iou_thresh_lo

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.bbox_reg_weights = bbox_reg_weights

    def forward(self, feature_maps: dict, proposals_list: list, image_shapes: list, targets: list = None):
        """
        feature_maps: dict of pyramid feature maps (p2..p5)
        proposals_list: List of proposal Tensors (N_prop, 4) per image
        image_shapes: List of (height, width) tuples
        targets: Optional ground-truth dicts for training
        """
        if self.training and targets is not None:
            # 1. Sample RoIs and generate training targets
            sampled_proposals, sampled_labels, sampled_deltas = self._select_training_samples(
                proposals_list, targets
            )
            # 2. Extract multi-scale pooled features for sampled RoIs
            roi_features = self.roi_align(feature_maps, sampled_proposals)
            cls_scores, bbox_deltas = self.box_head(roi_features)

            # 3. Compute RoI classification and box regression losses
            loss_cls, loss_reg = self._compute_losses(
                cls_scores, bbox_deltas, sampled_labels, sampled_deltas
            )
            return None, {"loss_roi_cls": loss_cls, "loss_roi_reg": loss_reg}
        else:
            # Inference mode
            roi_features = self.roi_align(feature_maps, proposals_list)
            cls_scores, bbox_deltas = self.box_head(roi_features)

            # Post-process into final detections per image
            detections = self._post_process(cls_scores, bbox_deltas, proposals_list, image_shapes)
            return detections, {}

    def _select_training_samples(self, proposals_list: list, targets: list):
        sampled_proposals = []
        sampled_labels = []
        sampled_deltas = []

        device = proposals_list[0].device if len(proposals_list) > 0 else torch.device("cpu")

        for b, props in enumerate(proposals_list):
            gt_boxes = targets[b]["boxes"].to(device)
            gt_labels = targets[b]["labels"].to(device)

            # Append GT boxes to proposals to guarantee high-quality positive anchors exist
            if gt_boxes.numel() > 0:
                all_props = torch.cat([props, gt_boxes], dim=0)
            else:
                all_props = props

            if all_props.numel() == 0:
                sampled_proposals.append(torch.empty((0, 4), device=device))
                sampled_labels.append(torch.empty((0,), dtype=torch.int64, device=device))
                sampled_deltas.append(torch.empty((0, 4), device=device))
                continue

            if gt_boxes.numel() == 0:
                # Background only
                neg_idx = torch.randperm(len(all_props), device=device)[:self.batch_size_per_image]
                sampled_proposals.append(all_props[neg_idx])
                sampled_labels.append(torch.full((len(neg_idx),), self.bg_class_idx, dtype=torch.int64, device=device))
                sampled_deltas.append(torch.zeros((len(neg_idx), 4), device=device))
                continue

            # Calculate IoU matrix (N_props, M_gt)
            iou_matrix = box_iou(all_props, gt_boxes)
            max_iou, best_gt_idx = iou_matrix.max(dim=1)

            # Assign labels
            pos_mask = max_iou >= self.fg_iou_thresh
            neg_mask = (max_iou >= self.bg_iou_thresh_lo) & (max_iou < self.bg_iou_thresh_hi)

            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]

            # Sample positive RoIs (up to 25%)
            max_pos = int(self.batch_size_per_image * self.positive_fraction)
            if len(pos_indices) > max_pos:
                perm = torch.randperm(len(pos_indices), device=device)
                pos_keep = pos_indices[perm[:max_pos]]
            else:
                pos_keep = pos_indices

            # Sample negative RoIs to fill remainder of batch
            max_neg = self.batch_size_per_image - len(pos_keep)
            if len(neg_indices) > max_neg:
                perm = torch.randperm(len(neg_indices), device=device)
                neg_keep = neg_indices[perm[:max_neg]]
            else:
                neg_keep = neg_indices

            keep = torch.cat([pos_keep, neg_keep])
            b_proposals = all_props[keep]

            # Labels for kept proposals
            b_labels = torch.full((len(keep),), self.bg_class_idx, dtype=torch.int64, device=device)
            if len(pos_keep) > 0:
                b_labels[:len(pos_keep)] = gt_labels[best_gt_idx[pos_keep]]

            # Targets deltas for positive proposals
            b_deltas = torch.zeros((len(keep), 4), device=device)
            if len(pos_keep) > 0:
                matched_gt = gt_boxes[best_gt_idx[pos_keep]]
                pos_props = all_props[pos_keep]
                b_deltas[:len(pos_keep)] = box_transform(pos_props, matched_gt, weights=self.bbox_reg_weights)

            sampled_proposals.append(b_proposals)
            sampled_labels.append(b_labels)
            sampled_deltas.append(b_deltas)

        return sampled_proposals, torch.cat(sampled_labels, dim=0), torch.cat(sampled_deltas, dim=0)

    def _compute_losses(self, cls_scores: torch.Tensor, bbox_deltas: torch.Tensor, labels: torch.Tensor, target_deltas: torch.Tensor):
        if len(labels) == 0:
            return torch.tensor(0.0, device=cls_scores.device), torch.tensor(0.0, device=cls_scores.device)

        # 1. Classification CrossEntropy Loss over (num_classes + 1)
        loss_cls = F.cross_entropy(cls_scores, labels)

        # 2. Regression Loss strictly on foreground positive RoIs
        pos_mask = labels < self.bg_class_idx
        num_pos = pos_mask.sum().item()

        if num_pos > 0:
            pos_labels = labels[pos_mask]
            # Select class-specific delta predictions: shape (N_pos, 4)
            pos_pred_deltas = bbox_deltas[pos_mask].view(-1, self.num_classes + 1, 4)
            selected_pred_deltas = pos_pred_deltas[torch.arange(num_pos, device=labels.device), pos_labels]

            loss_reg = F.smooth_l1_loss(selected_pred_deltas, target_deltas[pos_mask], beta=1.0, reduction="sum")
            loss_reg = loss_reg / max(float(len(labels)), 1.0)
        else:
            loss_reg = torch.tensor(0.0, device=cls_scores.device)

        return loss_cls, loss_reg

    @torch.no_grad()
    def _post_process(self, cls_scores: torch.Tensor, bbox_deltas: torch.Tensor, proposals_list: list, image_shapes: list):
        cls_probs = F.softmax(cls_scores, dim=-1)  # (N_total_rois, num_classes + 1)
        num_rois_per_img = [len(p) for p in proposals_list]

        split_probs = torch.split(cls_probs, num_rois_per_img)
        split_deltas = torch.split(bbox_deltas, num_rois_per_img)

        results = []
        for b, (probs, deltas, props, (img_h, img_w)) in enumerate(
            zip(split_probs, split_deltas, proposals_list, image_shapes)
        ):
            if len(props) == 0:
                results.append({
                    "boxes": torch.empty((0, 4), device=cls_scores.device),
                    "scores": torch.empty((0,), device=cls_scores.device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=cls_scores.device),
                })
                continue

            # (N_rois, num_classes + 1, 4)
            deltas = deltas.view(-1, self.num_classes + 1, 4)

            # Candidate detections across all foreground classes (0..num_classes-1)
            all_boxes = []
            all_scores = []
            all_labels = []

            for cls_idx in range(self.num_classes):
                c_probs = probs[:, cls_idx]
                c_deltas = deltas[:, cls_idx]

                keep = c_probs > self.score_thresh
                if not keep.any():
                    continue

                k_probs = c_probs[keep]
                k_deltas = c_deltas[keep]
                k_props = props[keep]

                k_boxes = box_decode(k_props, k_deltas, weights=self.bbox_reg_weights)
                k_boxes = clip_boxes(k_boxes, (img_h, img_w))

                # Filter invalid zero-area boxes
                valid = (k_boxes[:, 2] > k_boxes[:, 0] + 1) & (k_boxes[:, 3] > k_boxes[:, 1] + 1)
                k_boxes = k_boxes[valid]
                k_probs = k_probs[valid]

                if len(k_boxes) > 0:
                    k_labels = torch.full((len(k_boxes),), cls_idx, dtype=torch.int64, device=cls_scores.device)
                    all_boxes.append(k_boxes)
                    all_scores.append(k_probs)
                    all_labels.append(k_labels)

            if len(all_boxes) == 0:
                results.append({
                    "boxes": torch.empty((0, 4), device=cls_scores.device),
                    "scores": torch.empty((0,), device=cls_scores.device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=cls_scores.device),
                })
                continue

            cat_boxes = torch.cat(all_boxes, dim=0)
            cat_scores = torch.cat(all_scores, dim=0)
            cat_labels = torch.cat(all_labels, dim=0)

            # Apply Per-class NMS
            keep_indices = batched_nms(cat_boxes, cat_scores, cat_labels, self.nms_thresh)

            if len(keep_indices) > self.detections_per_img:
                keep_indices = keep_indices[:self.detections_per_img]

            results.append({
                "boxes": cat_boxes[keep_indices],
                "scores": cat_scores[keep_indices],
                "labels": cat_labels[keep_indices],
            })

        return results
