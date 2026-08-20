import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import roi_align as tv_roi_align
except ImportError:
    tv_roi_align = None


def pure_pytorch_roi_align(feature_map: torch.Tensor, rois: torch.Tensor, output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2):
    """
    Pure PyTorch RoIAlign fallback using torch.nn.functional.grid_sample (bilinear interpolation).
    feature_map: Tensor (B, C, H, W)
    rois: Tensor (N, 5) where each row is [batch_idx, x1, y1, x2, y2]
    output_size: (out_h, out_w)
    spatial_scale: scale factor from image coordinate to feature map coordinate
    """
    out_h, out_w = output_size
    n_rois = rois.shape[0]
    device = feature_map.device
    dtype = feature_map.dtype
    _, channels, feat_h, feat_w = feature_map.shape

    if n_rois == 0:
        return torch.empty((0, channels, out_h, out_w), device=device, dtype=dtype)

    batch_indices = rois[:, 0].long()
    scaled_rois = rois[:, 1:] * spatial_scale

    x1 = scaled_rois[:, 0]
    y1 = scaled_rois[:, 1]
    x2 = scaled_rois[:, 2]
    y2 = scaled_rois[:, 3]

    roi_w = torch.clamp(x2 - x1, min=1.0)
    roi_h = torch.clamp(y2 - y1, min=1.0)

    bin_size_w = roi_w / float(out_w)
    bin_size_h = roi_h / float(out_h)

    # Grid sampling points
    num_samples_w = sampling_ratio if sampling_ratio > 0 else math.ceil(bin_size_w.max().item())
    num_samples_h = sampling_ratio if sampling_ratio > 0 else math.ceil(bin_size_h.max().item())
    num_samples_w = max(1, min(int(num_samples_w), 4))
    num_samples_h = max(1, min(int(num_samples_h), 4))

    # Sample offsets within each bin
    sample_w = (torch.arange(num_samples_w, device=device, dtype=dtype) + 0.5) / num_samples_w
    sample_h = (torch.arange(num_samples_h, device=device, dtype=dtype) + 0.5) / num_samples_h

    # Bin origins
    bin_x = (torch.arange(out_w, device=device, dtype=dtype)[:, None] + sample_w[None, :]).view(-1)  # (out_w * num_samples_w)
    bin_y = (torch.arange(out_h, device=device, dtype=dtype)[:, None] + sample_h[None, :]).view(-1)  # (out_h * num_samples_h)

    # Coordinates for each RoI
    grid_x = x1[:, None] + bin_x[None, :] * (bin_size_w[:, None] / num_samples_w)
    grid_y = y1[:, None] + bin_y[None, :] * (bin_size_h[:, None] / num_samples_h)

    # Normalize to [-1, 1] for grid_sample
    grid_x_norm = (grid_x / max(float(feat_w - 1), 1.0)) * 2.0 - 1.0
    grid_y_norm = (grid_y / max(float(feat_h - 1), 1.0)) * 2.0 - 1.0

    # Expand grid
    grid = torch.stack([
        grid_x_norm.unsqueeze(1).expand(-1, grid_y_norm.shape[1], -1),
        grid_y_norm.unsqueeze(2).expand(-1, -1, grid_x_norm.shape[1]),
    ], dim=-1)  # (N, total_sample_h, total_sample_w, 2)

    # Sample features
    features_selected = feature_map[batch_indices]  # (N, C, H, W)
    sampled = F.grid_sample(features_selected, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Reshape and average pool over sample points inside each bin
    sampled = sampled.view(n_rois, channels, out_h, num_samples_h, out_w, num_samples_w)
    pooled = sampled.mean(dim=(3, 5))  # (N, C, out_h, out_w)

    return pooled


class MultiScaleRoIAlign(nn.Module):
    """
    Multi-Scale RoIAlign pooling across FPN levels (P2, P3, P4, P5).
    Maps RoIs to pyramid level based on standard FPN formula:
    k = floor(k0 + log2(sqrt(w*h) / 224)), clamped to [2, 5].
    """

    def __init__(self, output_size=(7, 7), sampling_ratio=2, canonical_level=4, canonical_scale=224):
        super().__init__()
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.canonical_level = canonical_level
        self.canonical_scale = canonical_scale
        self.spatial_scales = {
            "p2": 1.0 / 4.0,
            "p3": 1.0 / 8.0,
            "p4": 1.0 / 16.0,
            "p5": 1.0 / 32.0,
        }

    def _roi_align_op(self, feature_map: torch.Tensor, rois: torch.Tensor, spatial_scale: float) -> torch.Tensor:
        if tv_roi_align is not None:
            return tv_roi_align(
                feature_map,
                rois,
                output_size=self.output_size,
                spatial_scale=spatial_scale,
                sampling_ratio=self.sampling_ratio,
                aligned=True,
            )
        else:
            return pure_pytorch_roi_align(
                feature_map,
                rois,
                output_size=self.output_size,
                spatial_scale=spatial_scale,
                sampling_ratio=self.sampling_ratio,
            )

    def forward(self, feature_maps: dict, proposals_list: list) -> torch.Tensor:
        """
        feature_maps: dict containing 'p2', 'p3', 'p4', 'p5'
        proposals_list: List of Tensors (N_i, 4) in [xmin, ymin, xmax, ymax] per image in batch
        Returns: Tensor of pooled features (N_total_rois, C, out_h, out_w)
        """
        device = feature_maps["p2"].device
        channels = feature_maps["p2"].shape[1]
        out_h, out_w = self.output_size

        # Create rois tensor (N_total, 5) where [batch_idx, xmin, ymin, xmax, ymax]
        rois_with_batch = []
        for batch_idx, props in enumerate(proposals_list):
            if len(props) > 0:
                b_idx = torch.full((len(props), 1), batch_idx, dtype=props.dtype, device=device)
                rois_with_batch.append(torch.cat([b_idx, props], dim=1))

        if len(rois_with_batch) == 0:
            return torch.empty((0, channels, out_h, out_w), device=device)

        all_rois = torch.cat(rois_with_batch, dim=0)  # (N_total, 5)

        # Compute area to assign level
        w = (all_rois[:, 3] - all_rois[:, 1]).clamp(min=1.0)
        h = (all_rois[:, 4] - all_rois[:, 2]).clamp(min=1.0)
        scale = torch.sqrt(w * h)

        # FPN level assignment: k = floor(4 + log2(scale / 224))
        target_lvls = torch.floor(self.canonical_level + torch.log2(scale / float(self.canonical_scale) + 1e-6)).long()
        target_lvls = torch.clamp(target_lvls, min=2, max=5)

        pooled_levels = []
        original_indices = []

        for lvl in range(2, 6):
            key = f"p{lvl}"
            idx = torch.where(target_lvls == lvl)[0]
            if len(idx) > 0:
                lvl_rois = all_rois[idx]
                feat = feature_maps[key]
                scale_val = self.spatial_scales[key]
                pooled_lvl = self._roi_align_op(feat, lvl_rois, scale_val)
                pooled_levels.append(pooled_lvl)
                original_indices.append(idx)

        if len(pooled_levels) == 0:
            return torch.empty((0, channels, out_h, out_w), device=device, dtype=feature_maps["p2"].dtype)

        all_pooled = torch.cat(pooled_levels, dim=0)
        all_indices = torch.cat(original_indices, dim=0)

        # Restore original proposal order without in-place tensor mutations
        inv_indices = torch.empty_like(all_indices)
        inv_indices[all_indices] = torch.arange(len(all_indices), device=device)
        pooled_features = all_pooled[inv_indices]

        return pooled_features
