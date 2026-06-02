"""MVMS-MAE model: extends MVMS-v2 with lightweight MAE reconstruction head.

Architecture:
  - DGCNN point encoder (shared for all point views)
  - ResNet50 shared multi-level image encoder (same as MVMS-v2)
  - Point/image projectors for contrastive learning
  - PointMAEHead: MLP decoder for masked patch reconstruction
    Uses global feature from visible points + center positional embedding
    to reconstruct local_xyz of masked patches.

Key design choices:
  - MAE head uses the same DGCNN backbone to encode visible points
  - MLP decoder (not Transformer) for simplicity in v1
  - Center positional embedding to guide reconstruction
  - Chamfer distance loss on masked patches only
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

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


class SharedResNet50MultiLevelV2(nn.Module):
    """Shared ResNet50 with multi-level features for 4 image views.
    Identical to MVMS-v2 version.
    """

    def __init__(self, out_dim: int = 512):
        super().__init__()
        base = resnet50(weights=None)
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


class PointMAEHead(nn.Module):
    """Lightweight MLP decoder for masked patch reconstruction.

    Given global feature (from visible points) and masked patch center positions,
    reconstructs the local_xyz of each masked patch.

    Architecture:
      1. Positional MLP: center_xyz -> pos_embed [G, 256]
      2. Concatenate: [global_feat, pos_embed] -> [G, global_dim + 256]
      3. MLP decoder -> reconstructed local_xyz [G, group_size, 3]
    """

    def __init__(self, global_dim: int = 2048, hidden_dim: int = 512, group_size: int = 32):
        super().__init__()
        self.group_size = group_size

        # Positional embedding from center coordinates
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )

        # Decoder: global_feat + pos_embed -> local patch xyz
        self.decoder = nn.Sequential(
            nn.Linear(global_dim + 256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, group_size * 3),
        )

    def forward(self, global_feat: torch.Tensor, centers: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            global_feat: [B, global_dim] - global feature from visible points
            centers: [B, G, 3] - all patch center positions
            mask: [B, G] - 1=masked, 0=visible

        Returns:
            pred: [B, G, group_size, 3] - reconstructed local_xyz for ALL patches
                  (loss will only be computed on masked ones)
        """
        B, G, _ = centers.shape
        pos = self.pos_mlp(centers)  # [B, G, 256]
        glob = global_feat[:, None, :].expand(B, G, -1)  # [B, G, global_dim]
        token = torch.cat([glob, pos], dim=-1)  # [B, G, global_dim + 256]
        pred = self.decoder(token)  # [B, G, group_size * 3]
        pred = pred.reshape(B, G, self.group_size, 3)
        return pred


class MVMSMAECrossPoint(nn.Module):
    """MVMS-MAE: MVMS-v2 contrastive learning + MAE reconstruction.

    Components:
      - point_encoder: DGCNN (shared for all point views + MAE visible points)
      - point_projectors: per-view MLP projectors for contrastive learning
      - image_encoder: SharedResNet50MultiLevelV2
      - mae_head: PointMAEHead for masked patch reconstruction
    """

    def __init__(self, args, cfg: Dict):
        super().__init__()
        self.point_encoder = DGCNN(args)
        out_dim = cfg.get("projection_dim", 512)

        # Point projectors for contrastive views
        self.point_projectors = nn.ModuleDict({
            "clean_global": MLPProjector(args.emb_dims * 2, 1024, out_dim),
            "global": MLPProjector(args.emb_dims * 2, 1024, out_dim),
            "mid": MLPProjector(args.emb_dims * 2, 1024, out_dim),
            "local": MLPProjector(args.emb_dims * 2, 512, out_dim),
        })

        # MAE visible point projector (maps DGCNN output to MAE global feature)
        self.image_encoder = SharedResNet50MultiLevelV2(out_dim=out_dim)

        # MAE head
        mae_cfg = cfg.get("mae", {})
        self.mae_enabled = bool(mae_cfg.get("enabled", True))
        if self.mae_enabled:
            global_dim = int(mae_cfg.get("global_dim", 2048))
            hidden_dim = int(mae_cfg.get("hidden_dim", 512))
            group_size = int(mae_cfg.get("group_size", 32))
            self.mae_head = PointMAEHead(
                global_dim=global_dim,
                hidden_dim=hidden_dim,
                group_size=group_size,
            )

    def encode_point_tensor(self, points: torch.Tensor, view_name: str):
        x = points.transpose(2, 1).contiguous()
        _, _, _, feat = self.point_encoder(x)
        projected = self.point_projectors[view_name](feat)
        return {"feat": feat, "proj": projected}

    def encode_points(self, points: Dict[str, torch.Tensor]):
        return {name: self.encode_point_tensor(tensor, name) for name, tensor in points.items()}

    def encode_mae_visible(self, visible_points: torch.Tensor):
        """Encode visible points for MAE reconstruction.
        Returns the raw DGCNN global feature (2048-dim).
        """
        x = visible_points.transpose(2, 1).contiguous()
        _, _, _, feat = self.point_encoder(x)
        return feat  # [B, 2048]

    def forward(self, batch):
        result = {
            "points": self.encode_points(batch["points"]),
            "images": self.image_encoder(batch["images"]),
        }

        # MAE forward pass
        if self.mae_enabled and "mae" in batch and len(batch["mae"]) > 0:
            visible_pts = batch["mae"]["visible_points"]  # [B, V, 3]
            centers = batch["mae"]["centers"]              # [B, G, 3]
            mask = batch["mae"]["mask"]                     # [B, G]
            target_local_xyz = batch["mae"]["local_xyz"]   # [B, G, S, 3]

            # Encode visible points to get global feature
            global_feat = self.encode_mae_visible(visible_pts)  # [B, 2048]

            # Reconstruct all patches (loss will mask only masked ones)
            pred_local_xyz = self.mae_head(global_feat, centers, mask)  # [B, G, S, 3]

            result["mae"] = {
                "pred": pred_local_xyz,
                "target": target_local_xyz,
                "mask": mask,
                "global_feat": global_feat,
            }

        return result
