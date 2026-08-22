import torch
import torch.nn as nn
from models.backbone import ResNetBackbone
from models.fpn import FeaturePyramidNetwork
from models.rpn import RegionProposalNetwork
from models.roi_head import RoIHeads


class FasterRCNN(nn.Module):
    """
    Faster R-CNN Object Detector written completely from scratch in PyTorch.
    Architecture:
      - ResNet Backbone (C2, C3, C4, C5)
      - Feature Pyramid Network (P2, P3, P4, P5, P6)
      - Region Proposal Network (Anchor Generation, RPN Head, RPN Matcher & Sampler)
      - Multi-Scale RoIAlign (P2-P5 feature extraction)
      - Fast R-CNN 2-FC Box Head (Class-specific box regression & (K+1)-class classification)
    """

    def __init__(
        self,
        num_classes=5,
        backbone_name="resnet50",
        pretrained=True,
        freeze_bn=True,
        ratios=(0.5, 1.0, 2.0),
        scales=(1.0,),
        conf_threshold=0.05,
        nms_threshold=0.5,
        max_detections_per_img=100,
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=1000,
        fc_dim=1024,
        dropout_p=0.0,
        roi_batch_size_per_image=512,
        roi_positive_fraction=0.25,
        box_loss_type="smooth_l1",
        box_loss_weight=1.0,
        use_soft_nms=False,
        soft_nms_sigma=0.5,
        soft_nms_method="gaussian",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_detections_per_img = max_detections_per_img

        # 1. Backbone Feature Extractor (with FrozenBatchNorm2d for stable small-batch fine-tuning)
        self.backbone = ResNetBackbone(architecture=backbone_name, pretrained=pretrained, freeze_bn=freeze_bn)

        # 2. Feature Pyramid Network (FPN)
        self.fpn = FeaturePyramidNetwork(in_channels_list=self.backbone.out_channels, out_channels=256)

        # 3. Region Proposal Network (RPN)
        self.rpn = RegionProposalNetwork(
            in_channels=256,
            ratios=ratios,
            scales=scales,
            rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            rpn_nms_thresh=0.7,
            rpn_batch_size_per_image=256,
            rpn_positive_fraction=0.5,
            box_loss_type=box_loss_type,
        )

        # 4. RoI Head (Multi-Scale RoIAlign + Fast R-CNN 2-FC Head with Dropout)
        self.roi_heads = RoIHeads(
            in_channels=256,
            num_classes=num_classes,
            roi_size=(7, 7),
            fc_dim=fc_dim,
            dropout_p=dropout_p,
            batch_size_per_image=roi_batch_size_per_image,
            positive_fraction=roi_positive_fraction,
            fg_iou_thresh=0.5,
            bg_iou_thresh_hi=0.5,
            bg_iou_thresh_lo=0.0,
            score_thresh=conf_threshold,
            nms_thresh=nms_threshold,
            detections_per_img=max_detections_per_img,
            box_loss_type=box_loss_type,
            box_loss_weight=box_loss_weight,
            use_soft_nms=use_soft_nms,
            soft_nms_sigma=soft_nms_sigma,
            soft_nms_method=soft_nms_method,
        )

    def forward(self, images: torch.Tensor, targets: list = None):
        """
        images: Tensor (B, 3, H, W)
        targets: Optional list of dicts with 'boxes' (N, 4) and 'labels' (N,)
        """
        batch_size, _, img_h, img_w = images.shape
        image_shapes = [(img_h, img_w)] * batch_size

        # 1. Extract multi-level features
        backbone_feats = self.backbone(images)
        fpn_feats = self.fpn(backbone_feats)

        if self.training and targets is not None:
            # 2. RPN proposals and losses
            proposals, rpn_losses = self.rpn(fpn_feats, (img_h, img_w), targets)

            # 3. RoI head losses
            _, roi_losses = self.roi_heads(fpn_feats, proposals, image_shapes, targets)

            # Total loss
            loss_rpn_cls = rpn_losses["loss_rpn_cls"]
            loss_rpn_reg = rpn_losses["loss_rpn_reg"]
            loss_roi_cls = roi_losses["loss_roi_cls"]
            loss_roi_reg = roi_losses["loss_roi_reg"]

            total_loss = loss_rpn_cls + loss_rpn_reg + loss_roi_cls + loss_roi_reg

            return {
                "loss": total_loss,
                "loss_rpn_cls": loss_rpn_cls,
                "loss_rpn_reg": loss_rpn_reg,
                "loss_roi_cls": loss_roi_cls,
                "loss_roi_reg": loss_roi_reg,
            }
        else:
            # Inference mode
            proposals, _ = self.rpn(fpn_feats, (img_h, img_w))
            detections, _ = self.roi_heads(fpn_feats, proposals, image_shapes)
            return detections
