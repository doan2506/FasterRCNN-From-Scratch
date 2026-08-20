import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import box_iou, box_transform, box_decode, clip_boxes
from utils.nms import nms


class RPNAnchorGenerator(nn.Module):
    """
    Generates multi-scale anchor boxes across pyramid levels (P2, P3, P4, P5, P6).
    """

    def __init__(
        self,
        strides=(4, 8, 16, 32, 64),
        base_sizes=(32, 64, 128, 256, 512),
        ratios=(0.33, 0.5, 1.0, 2.0, 3.0),
    ):
        super().__init__()
        self.strides = strides
        self.base_sizes = base_sizes
        self.register_buffer("ratios", torch.tensor(ratios, dtype=torch.float32))
        self.num_anchors_per_location = len(ratios)

    def _generate_base_anchors(self, base_size: float, device: torch.device) -> torch.Tensor:
        """
        Generates base anchors centered at (0, 0) for a given base size across ratios.
        """
        ratios = self.ratios
        aspect_ratios = torch.sqrt(ratios)
        h = base_size / aspect_ratios
        w = base_size * aspect_ratios

        xmin = -0.5 * w
        ymin = -0.5 * h
        xmax = 0.5 * w
        ymax = 0.5 * h

        return torch.stack([xmin, ymin, xmax, ymax], dim=1)

    def forward(self, feature_maps: dict, image_size: tuple) -> torch.Tensor:
        """
        feature_maps: dict with keys 'p2', 'p3', 'p4', 'p5', 'p6'
        image_size: (height, width)
        Returns: Tensor of shape (N_total, 4) in [xmin, ymin, xmax, ymax]
        """
        device = feature_maps["p2"].device
        all_anchors = []

        level_keys = ["p2", "p3", "p4", "p5", "p6"]
        for key, stride, base_size in zip(level_keys, self.strides, self.base_sizes):
            feat = feature_maps[key]
            _, _, feat_h, feat_w = feat.shape
            base_anchors = self._generate_base_anchors(base_size, device)  # (3, 4)

            shift_x = (torch.arange(0, feat_w, device=device, dtype=torch.float32) + 0.5) * stride
            shift_y = (torch.arange(0, feat_h, device=device, dtype=torch.float32) + 0.5) * stride

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shifts = torch.stack(
                [shift_x.reshape(-1), shift_y.reshape(-1), shift_x.reshape(-1), shift_y.reshape(-1)],
                dim=1,
            )

            # (H*W, 1, 4) + (1, 3, 4) -> (H*W, 3, 4) -> (H*W*3, 4)
            anchors = (shifts[:, None, :] + base_anchors[None, :, :]).reshape(-1, 4)
            all_anchors.append(anchors)

        return torch.cat(all_anchors, dim=0)


class RPNHead(nn.Module):
    """
    Standard RPN Head shared across all pyramid levels.
    """

    def __init__(self, in_channels=256, num_anchors=5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feature_maps: dict) -> tuple:
        level_keys = ["p2", "p3", "p4", "p5", "p6"]
        cls_logits_list = []
        bbox_pred_list = []

        for key in level_keys:
            feat = feature_maps[key]
            batch_size = feat.shape[0]
            t = F.relu(self.conv(feat), inplace=False)

            logits = self.cls_logits(t)  # (B, A*1, H, W)
            deltas = self.bbox_pred(t)   # (B, A*4, H, W)

            # Permute & reshape
            logits = logits.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
            deltas = deltas.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

            cls_logits_list.append(logits)
            bbox_pred_list.append(deltas)

        all_cls_logits = torch.cat(cls_logits_list, dim=1)  # (B, N_total, 1)
        all_bbox_pred = torch.cat(bbox_pred_list, dim=1)    # (B, N_total, 4)

        return all_cls_logits, all_bbox_pred


class RegionProposalNetwork(nn.Module):
    """
    Region Proposal Network (RPN) for Faster R-CNN.
    Handles anchor generation, proposal generation, target matching, and RPN losses.
    """

    def __init__(
        self,
        in_channels=256,
        ratios=(0.33, 0.5, 1.0, 2.0, 3.0),
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
    ):
        super().__init__()
        self.anchor_generator = RPNAnchorGenerator(ratios=ratios)
        self.head = RPNHead(in_channels=in_channels, num_anchors=self.anchor_generator.num_anchors_per_location)

        self.pre_nms_top_n_train = rpn_pre_nms_top_n_train
        self.post_nms_top_n_train = rpn_post_nms_top_n_train
        self.pre_nms_top_n_test = rpn_pre_nms_top_n_test
        self.post_nms_top_n_test = rpn_post_nms_top_n_test
        self.nms_thresh = rpn_nms_thresh
        self.batch_size_per_image = rpn_batch_size_per_image
        self.positive_fraction = rpn_positive_fraction

    def forward(self, feature_maps: dict, image_shape: tuple, targets: list = None):
        """
        feature_maps: dict of pyramid feature maps (p2..p6)
        image_shape: (height, width)
        targets: Optional list of target dicts for training
        Returns:
            proposals: List of Tensors (N_prop, 4) per image
            losses: dict containing 'loss_rpn_cls' and 'loss_rpn_reg' (when targets is provided)
        """
        img_h, img_w = image_shape
        anchors = self.anchor_generator(feature_maps, image_shape)  # (N_total, 4)
        cls_logits, bbox_deltas = self.head(feature_maps)           # (B, N_total, 1), (B, N_total, 4)

        batch_size = cls_logits.shape[0]

        # Generate candidate proposals (discrete candidate sampling)
        with torch.no_grad():
            pre_nms_top_n = self.pre_nms_top_n_train if self.training else self.pre_nms_top_n_test
            post_nms_top_n = self.post_nms_top_n_train if self.training else self.post_nms_top_n_test

            proposals = []
            cls_probs = torch.sigmoid(cls_logits.squeeze(-1))  # (B, N_total)

            for b in range(batch_size):
                b_probs = cls_probs[b]
                b_deltas = bbox_deltas[b]

                # Decode proposals from anchors
                b_boxes = box_decode(anchors, b_deltas)
                b_boxes = clip_boxes(b_boxes, (img_h, img_w))

                # Filter minimal size boxes (min 1px)
                ws = b_boxes[:, 2] - b_boxes[:, 0]
                hs = b_boxes[:, 3] - b_boxes[:, 1]
                valid_size = (ws >= 1.0) & (hs >= 1.0)

                valid_boxes = b_boxes[valid_size]
                valid_probs = b_probs[valid_size]

                # Select top pre_nms proposals
                if len(valid_probs) > pre_nms_top_n:
                    topk_probs, topk_idx = torch.topk(valid_probs, pre_nms_top_n)
                    valid_boxes = valid_boxes[topk_idx]
                    valid_probs = topk_probs

                # Apply NMS
                keep = nms(valid_boxes, valid_probs, self.nms_thresh)
                keep = keep[:post_nms_top_n]
                b_proposals = valid_boxes[keep]

                proposals.append(b_proposals)

        losses = {}
        if self.training and targets is not None:
            loss_cls, loss_reg = self._compute_losses(anchors, cls_logits, bbox_deltas, targets)
            losses["loss_rpn_cls"] = loss_cls
            losses["loss_rpn_reg"] = loss_reg

        return proposals, losses

    def _compute_losses(self, anchors: torch.Tensor, cls_logits: torch.Tensor, bbox_deltas: torch.Tensor, targets: list):
        batch_size = cls_logits.shape[0]
        device = cls_logits.device
        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        total_positives = 0

        for b in range(batch_size):
            gt_boxes = targets[b]["boxes"].to(device)
            if gt_boxes.numel() == 0:
                # Background-only image: sample 256 negative anchors
                neg_idx = torch.randperm(len(anchors), device=device)[:self.batch_size_per_image]
                neg_logits = cls_logits[b, neg_idx, 0]
                neg_targets = torch.zeros_like(neg_logits)
                total_cls_loss += F.binary_cross_entropy_with_logits(neg_logits, neg_targets, reduction="sum") / self.batch_size_per_image
                continue

            iou_matrix = box_iou(anchors, gt_boxes)  # (N_anchors, M_gt)
            max_iou_per_anchor, best_gt_idx = iou_matrix.max(dim=1)

            # Initialize labels: -1 = ignore, 0 = bg, 1 = fg
            labels = torch.full((len(anchors),), -1, dtype=torch.float32, device=device)

            # Negative: IoU < 0.3
            labels[max_iou_per_anchor < 0.3] = 0.0

            # Positive: IoU >= 0.7
            labels[max_iou_per_anchor >= 0.7] = 1.0

            # Force at least one positive anchor per GT box (max IoU anchor)
            max_iou_per_gt, best_anchor_per_gt = iou_matrix.max(dim=0)
            for gt_i, anchor_i in enumerate(best_anchor_per_gt):
                labels[anchor_i] = 1.0
                best_gt_idx[anchor_i] = gt_i

            # Sample balanced batch (positive fraction 0.5 up to 256 total)
            pos_indices = torch.where(labels == 1.0)[0]
            neg_indices = torch.where(labels == 0.0)[0]

            max_pos = int(self.batch_size_per_image * self.positive_fraction)
            if len(pos_indices) > max_pos:
                perm = torch.randperm(len(pos_indices), device=device)
                pos_keep = pos_indices[perm[:max_pos]]
            else:
                pos_keep = pos_indices

            max_neg = self.batch_size_per_image - len(pos_keep)
            if len(neg_indices) > max_neg:
                perm = torch.randperm(len(neg_indices), device=device)
                neg_keep = neg_indices[perm[:max_neg]]
            else:
                neg_keep = neg_indices

            sampled_indices = torch.cat([pos_keep, neg_keep])
            if len(sampled_indices) == 0:
                continue

            # 1. RPN Classification Loss (BCE)
            sampled_logits = cls_logits[b, sampled_indices, 0]
            sampled_targets = labels[sampled_indices]
            b_cls_loss = F.binary_cross_entropy_with_logits(sampled_logits, sampled_targets, reduction="sum")
            total_cls_loss += b_cls_loss / max(len(sampled_indices), 1)

            # 2. RPN Regression Loss (Smooth L1 on positive anchors)
            if len(pos_keep) > 0:
                pos_anchors = anchors[pos_keep]
                pos_gt_boxes = gt_boxes[best_gt_idx[pos_keep]]
                target_deltas = box_transform(pos_anchors, pos_gt_boxes)
                pred_pos_deltas = bbox_deltas[b, pos_keep]

                b_reg_loss = F.smooth_l1_loss(pred_pos_deltas, target_deltas, beta=1.0 / 9.0, reduction="sum")
                total_reg_loss += b_reg_loss / max(len(pos_keep), 1)
                total_positives += len(pos_keep)

        loss_rpn_cls = total_cls_loss / batch_size
        loss_rpn_reg = total_reg_loss / batch_size

        return loss_rpn_cls, loss_rpn_reg
