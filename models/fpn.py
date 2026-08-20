import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN) generating multi-scale feature maps (P2, P3, P4, P5, P6).
    P2: Stride 4  (high resolution for small objects like cups, bottles)
    P3: Stride 8
    P4: Stride 16
    P5: Stride 32
    P6: Stride 64 (subsampled from P5 for large RPN proposals)
    """

    def __init__(self, in_channels_list: dict, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        # 1x1 Conv Lateral connections
        self.lat_c5 = nn.Conv2d(in_channels_list["c5"], out_channels, kernel_size=1)
        self.lat_c4 = nn.Conv2d(in_channels_list["c4"], out_channels, kernel_size=1)
        self.lat_c3 = nn.Conv2d(in_channels_list["c3"], out_channels, kernel_size=1)
        self.lat_c2 = nn.Conv2d(in_channels_list["c2"], out_channels, kernel_size=1)

        # 3x3 Conv Smooth layers for P2, P3, P4, P5
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # P6: MaxPool stride 2 on P5 for RPN
        self.p6_pool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs: dict) -> dict:
        c2, c3, c4, c5 = inputs["c2"], inputs["c3"], inputs["c4"], inputs["c5"]

        # Top-down pathway
        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        # Smooth layers
        p2 = self.smooth_p2(p2)
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)

        # P6 for RPN
        p6 = self.p6_pool(p5)

        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6}
