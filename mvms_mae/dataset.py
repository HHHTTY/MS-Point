"""MVMS-MAE dataset: extends MVMS-v2 dataset with MAE patch grouping.

Point views: clean_global, global, mid, local (same as MVMS-v2)
Image views: weak_full, full, crop, hard (same as MVMS-v2)
MAE branch: groups clean_global into patches with masking
"""

import glob
import os
import random
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from datasets.plyfile import load_ply
from .augmentations import (
    PointViewTransform, build_image_transform,
    farthest_point_sample_np, pc_normalize_np,
    fps_torch, knn_torch, group_points_with_centers,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_shapenet_paths(root: str = "data/ShapeNet") -> List[str]:
    paths = []
    for cls in glob.glob(os.path.join(root, "*")):
        paths.extend(glob.glob(os.path.join(cls, "*.ply")))
    return sorted(paths)


def get_render_imgs(pcd_path: str) -> List[str]:
    parts = pcd_path.split("/")
    parts[1] = "ShapeNetRendering"
    parts[-1] = parts[-1][:-4]
    parts.append("rendering")
    return sorted(glob.glob(os.path.join("/".join(parts), "*.png")))


def prepare_mae_patches_np(
    raw_points: np.ndarray,
    num_centers: int = 64,
    group_size: int = 32,
    mask_ratio: float = 0.5,
    visible_num_points: int = 1024,
    normalize_patch: bool = True,
):
    """Prepare MAE patches from raw point cloud using NumPy (CPU-side).

    Steps:
    1. Normalize + FPS sample to get sufficient points
    2. FPS select num_centers patch centers
    3. kNN find group_size neighbors per center -> local_xyz
    4. Optionally normalize each patch to center-local coords
    5. Random mask: keep (1 - mask_ratio) patches as visible
    6. Collect visible points for the encoder

    Returns dict with torch tensors ready for the model.
    """
    pts = raw_points.astype(np.float32, copy=True)
    pts = pc_normalize_np(pts)

    # Sample enough points for grouping (need at least num_centers * group_size)
    total_needed = max(visible_num_points, num_centers * group_size)
    total_needed = max(total_needed, 2048)
    pts = farthest_point_sample_np(pts, total_needed)

    # FPS for patch centers
    xyz = pts[:, :3].astype(np.float32)
    N = xyz.shape[0]
    selected_centers = np.empty(num_centers, dtype=np.int64)
    distances = np.full(N, 1e10, dtype=np.float32)
    farthest = np.random.randint(0, N)
    for i in range(num_centers):
        selected_centers[i] = farthest
        centroid = xyz[farthest]
        dist = ((xyz - centroid) ** 2).sum(axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(distances.argmax())

    centers_xyz = xyz[selected_centers]  # [G, 3]

    # kNN grouping: for each center, find group_size nearest neighbors
    local_patches = np.zeros((num_centers, group_size, 3), dtype=np.float32)
    for i in range(num_centers):
        c = xyz[selected_centers[i]]
        d = ((xyz - c) ** 2).sum(axis=1)
        nn_idx = np.argpartition(d, group_size)[:group_size]
        patch_pts = xyz[nn_idx]  # [group_size, 3]
        if normalize_patch:
            patch_pts = patch_pts - c  # center-local coordinates
        local_patches[i] = patch_pts

    # Random masking
    num_masked = int(num_centers * mask_ratio)
    num_visible = num_centers - num_masked
    perm = np.random.permutation(num_centers)
    visible_idx = perm[:num_visible]
    masked_idx = perm[num_visible:]

    # Collect visible points for encoder input (exact visible_num_points via FPS)
    # Use all points, sample exactly visible_num_points using FPS
    from .augmentations import farthest_point_sample_np as _fps
    visible_points = _fps(pts[:, :3].astype(np.float32), visible_num_points)[:, :3].astype(np.float32)

    # Create mask: 1 = masked, 0 = visible
    mask = np.zeros(num_centers, dtype=np.float32)
    mask[masked_idx] = 1.0

    return {
        "visible_points": torch.from_numpy(visible_points),        # [V, 3]
        "centers": torch.from_numpy(centers_xyz),                   # [G, 3]
        "local_xyz": torch.from_numpy(local_patches),              # [G, group_size, 3]
        "mask": torch.from_numpy(mask),                             # [G]
    }


class ShapeNetMVMSMAE(Dataset):
    """MVMS-MAE dataset: MVMS-v2 views + MAE patch grouping."""

    def __init__(self, cfg: Dict, max_samples: int = None):
        self.cfg = cfg
        self.data = load_shapenet_paths(cfg.get("point_root", "data/ShapeNet"))
        if max_samples is not None:
            self.data = self.data[:max_samples]
        if not self.data:
            raise RuntimeError("No ShapeNet point clouds found")

        self.point_transforms = {
            name: PointViewTransform(view_cfg)
            for name, view_cfg in cfg["point_views"].items()
        }
        self.image_transforms = {
            name: build_image_transform(view_cfg.get("kind", name), cfg.get("image_size", 224))
            for name, view_cfg in cfg["image_views"].items()
        }
        self.same_base_image = bool(cfg.get("same_base_image", True))

        # MAE config
        mae_cfg = cfg.get("mae", {})
        self.mae_enabled = bool(mae_cfg.get("enabled", True))
        self.mae_num_centers = int(mae_cfg.get("num_groups", 64))
        self.mae_group_size = int(mae_cfg.get("group_size", 32))
        self.mae_mask_ratio = float(mae_cfg.get("mask_ratio", 0.5))
        self.mae_visible_num_points = int(mae_cfg.get("visible_num_points", 1024))
        self.mae_normalize_patch = bool(mae_cfg.get("normalize_patch", True))
        # Range for random mask ratio
        self.mae_mask_ratio_min = float(mae_cfg.get("mask_ratio_min", 0.3))
        self.mae_mask_ratio_max = float(mae_cfg.get("mask_ratio_max", 0.6))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pcd_path = self.data[idx]
        point_cloud = load_ply(pcd_path)
        render_paths = get_render_imgs(pcd_path)
        if len(render_paths) < 1:
            raise RuntimeError(f"No render images for {pcd_path}")

        # Point views (same as MVMS-v2)
        points = {
            name: transform(point_cloud.copy())
            for name, transform in self.point_transforms.items()
        }

        # Image views (same as MVMS-v2)
        images = {}
        if self.same_base_image:
            img_path = random.choice(render_paths)
            base_img = Image.open(img_path).convert("RGB")
            for name, transform in self.image_transforms.items():
                images[name] = transform(base_img.copy())
        else:
            n_views = len(self.image_transforms)
            chosen = random.sample(render_paths, min(n_views, len(render_paths)))
            while len(chosen) < n_views:
                chosen.append(random.choice(render_paths))
            for (name, transform), img_path in zip(self.image_transforms.items(), chosen):
                images[name] = transform(Image.open(img_path).convert("RGB"))

        # MAE branch
        mae = {}
        if self.mae_enabled:
            # Use the raw (un-augmented) point cloud for MAE target
            # Randomize mask ratio within range
            mr = random.uniform(self.mae_mask_ratio_min, self.mae_mask_ratio_max)
            mae = prepare_mae_patches_np(
                point_cloud.copy(),
                num_centers=self.mae_num_centers,
                group_size=self.mae_group_size,
                mask_ratio=mr,
                visible_num_points=self.mae_visible_num_points,
                normalize_patch=self.mae_normalize_patch,
            )

        return {"points": points, "images": images, "mae": mae, "path": pcd_path}
