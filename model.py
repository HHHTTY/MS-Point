"""MVMS-v2 model with ImageNet pretrained ResNet50.

Identical to mvms_v2_improved_experiments/model.py except:
  - ResNet50 uses ImageNet pretrained weights instead of random initialization
  - Expected benefit: better initial image features → higher MN40 accuracy
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from models.dgcnn import DGCNN


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 1024, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class SharedResNet50MultiLevelV2Pretrained(nn.Module):
    """Shared ResNet50 with ImageNet pretrained weights for 4 image views.

    Key difference from SharedResNet50MultiLevelV2:
      - Uses ImageNet pretrained weights instead of random init
      - The pretrained backbone provides strong initial visual features
    """

    def __init__(self, out_dim: int = 512):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.projectors = nn.ModuleDict({
            "weak_full": MLPProjector(2048, 1024, out_dim),
            "full": MLPProjector(2048, 1024, out_dim),
            "crop": MLPProjector(1024, 1024, out_dim),
            "hard": MLPProjector(512, 512, out_dim),
        })

    def forward_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return {
            "hard": F.adaptive_avg_pool2d(l2, 1).flatten(1),
            "crop": F.adaptive_avg_pool2d(l3, 1).flatten(1),
            "full": F.adaptive_avg_pool2d(l4, 1).flatten(1),
        }

    def forward(self, images: Dict[str, torch.Tensor]):
        outputs = {}
        feat_cache = {}

        for name, projector in self.projectors.items():
            if name not in images:
                continue
            img = images[name]
            if name in ("weak_full", "full"):
                level = "full"
            elif name == "crop":
                level = "crop"
            else:
                level = "hard"

            if level not in feat_cache:
                feat_cache[level] = self.forward_features(img)
            outputs[name] = projector(feat_cache[level][level])
        return outputs


class MVMSv2CrossPointPretrained(nn.Module):
    """MVMS-v2 model with ImageNet pretrained ResNet50."""

    def __init__(self, args, cfg: Dict):
        super().__init__()
        self.point_encoder = DGCNN(args)
        out_dim = cfg.get("projection_dim", 512)

        self.point_projectors = nn.ModuleDict({
            "clean_global": MLPProjector(args.emb_dims * 2, 1024, out_dim),
            "global": MLPProjector(args.emb_dims * 2, 1024, out_dim),
            "mid": MLPProjector(args.emb_dims * 2, 1024, out_dim),
            "local": MLPProjector(args.emb_dims * 2, 512, out_dim),
        })

        self.image_encoder = SharedResNet50MultiLevelV2Pretrained(out_dim=out_dim)

    def encode_point_tensor(self, points: torch.Tensor, view_name: str):
        x = points.transpose(2, 1).contiguous()
        _, _, _, feat = self.point_encoder(x)
        projected = self.point_projectors[view_name](feat)
        return {"feat": feat, "proj": projected}

    def encode_points(self, points: Dict[str, torch.Tensor]):
        return {name: self.encode_point_tensor(tensor, name) for name, tensor in points.items()}

    def forward(self, batch):
        return {
            "points": self.encode_points(batch["points"]),
            "images": self.image_encoder(batch["images"]),
        }
