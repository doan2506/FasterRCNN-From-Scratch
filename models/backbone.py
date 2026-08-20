import torch
import torch.nn as nn
import torchvision.models as models


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and affine parameters are fixed.
    Standard for Object Detection backbones to prevent noisy statistics
    with small batch sizes and avoid polluting stats during validation.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class ResNetBackbone(nn.Module):
    """
    ResNet Feature Extractor returning multi-level feature maps C2, C3, C4, C5.
    Uses FrozenBatchNorm2d to stabilize training with small batch sizes.
    Supports ResNet-18, ResNet-34, and ResNet-50 backbones.
    """

    def __init__(self, architecture="resnet50", pretrained=True, freeze_bn=True):
        super().__init__()
        self.architecture = architecture
        norm_layer = FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d

        if architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights, norm_layer=norm_layer)
            self.out_channels = {"c2": 64, "c3": 128, "c4": 256, "c5": 512}
        elif architecture == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights, norm_layer=norm_layer)
            self.out_channels = {"c2": 64, "c3": 128, "c4": 256, "c5": 512}
        elif architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights, norm_layer=norm_layer)
            self.out_channels = {"c2": 256, "c3": 512, "c4": 1024, "c5": 2048}
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Stem layers (conv1 -> bn1 -> relu -> maxpool)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        # ResNet Residual Blocks
        self.layer1 = resnet.layer1  # C2 (stride 4)
        self.layer2 = resnet.layer2  # C3 (stride 8)
        self.layer3 = resnet.layer3  # C4 (stride 16)
        self.layer4 = resnet.layer4  # C5 (stride 32)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}
