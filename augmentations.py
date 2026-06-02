"""MVMS-v2 augmentations: extends MVMS augmentations with clean_global point transform and weak_full image transform.

clean_global: minimal augmentation (weak scale, tiny jitter, near-zero dropout)
weak_full:    full-size image with very mild augmentation (no crop, minimal jitter)
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def pc_normalize_np(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32, copy=True)
    centroid = pts[:, :3].mean(axis=0)
    pts[:, :3] -= centroid
    radius = np.sqrt((pts[:, :3] ** 2).sum(axis=1)).max()
    if radius > 1e-6:
        pts[:, :3] /= radius
    return pts


def farthest_point_sample_np(points: np.ndarray, num_points: int) -> np.ndarray:
    n = points.shape[0]
    if n == 0:
        raise ValueError("Cannot sample from an empty point cloud")
    if n <= num_points:
        idx = np.random.choice(n, num_points, replace=True)
        return points[idx].astype(np.float32)

    xyz = points[:, :3].astype(np.float32)
    selected = np.empty(num_points, dtype=np.int64)
    distances = np.full(n, 1e10, dtype=np.float32)
    farthest = np.random.randint(0, n)
    for i in range(num_points):
        selected[i] = farthest
        centroid = xyz[farthest]
        dist = ((xyz - centroid) ** 2).sum(axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(distances.argmax())
    return points[selected].astype(np.float32)


def random_resample_np(points: np.ndarray, num_points: int) -> np.ndarray:
    n = points.shape[0]
    if n == 0:
        raise ValueError("Cannot resample from an empty point cloud")
    replace = n < num_points
    idx = np.random.choice(n, num_points, replace=replace)
    return points[idx].astype(np.float32)


def rotate_y_np(points: np.ndarray) -> np.ndarray:
    angle = np.random.uniform(0.0, 2.0 * math.pi)
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    pts = points.copy()
    pts[:, :3] = pts[:, :3] @ rot.T
    return pts


def scale_np(points: np.ndarray, lo: float, hi: float) -> np.ndarray:
    pts = points.copy()
    pts[:, :3] *= np.random.uniform(lo, hi)
    return pts


def translate_np(points: np.ndarray, translate_range: float) -> np.ndarray:
    pts = points.copy()
    coord_min = pts[:, :3].min(axis=0)
    coord_max = pts[:, :3].max(axis=0)
    shift = np.random.uniform(-translate_range, translate_range, size=3).astype(np.float32) * (coord_max - coord_min)
    pts[:, :3] += shift
    return pts


def jitter_np(points: np.ndarray, std: float, clip: float) -> np.ndarray:
    pts = points.copy()
    noise = np.clip(np.random.normal(0.0, std, size=(pts.shape[0], 3)), -clip, clip).astype(np.float32)
    pts[:, :3] += noise
    return pts


def plane_crop_np(points: np.ndarray, remove_ratio: float) -> np.ndarray:
    xyz = points[:, :3]
    direction = np.random.normal(size=3).astype(np.float32)
    direction /= np.linalg.norm(direction) + 1e-8
    scores = xyz @ direction
    q = np.quantile(scores, remove_ratio)
    kept = points[scores >= q]
    return kept if kept.shape[0] > 8 else points


def sphere_crop_np(points: np.ndarray, remove_ratio: float) -> np.ndarray:
    xyz = points[:, :3]
    center = xyz[np.random.randint(0, xyz.shape[0])]
    dist = ((xyz - center) ** 2).sum(axis=1)
    remove_n = max(1, min(points.shape[0] - 8, int(points.shape[0] * remove_ratio)))
    remove_idx = np.argpartition(dist, remove_n)[:remove_n]
    mask = np.ones(points.shape[0], dtype=bool)
    mask[remove_idx] = False
    kept = points[mask]
    return kept if kept.shape[0] > 8 else points


def density_dropout_np(points: np.ndarray, drop_ratio: float) -> np.ndarray:
    n = points.shape[0]
    keep_n = max(8, int(n * (1.0 - drop_ratio)))
    idx = np.random.choice(n, keep_n, replace=False)
    return points[idx]


def view_occlusion_np(points: np.ndarray, remove_ratio: float) -> np.ndarray:
    xyz = points[:, :3]
    view = np.random.normal(size=3).astype(np.float32)
    view /= np.linalg.norm(view) + 1e-8
    depth = xyz @ view
    q = np.quantile(depth, remove_ratio)
    kept = points[depth >= q]
    return kept if kept.shape[0] > 8 else points


@dataclass
class PointViewConfig:
    name: str
    num_points: int
    sample: str = "fps"
    rotate: bool = True
    scale: Tuple[float, float] = (0.8, 1.25)
    translate: float = 0.1
    jitter_std: float = 0.01
    jitter_clip: float = 0.05
    plane_crop_prob: float = 0.0
    sphere_crop_prob: float = 0.0
    view_occlusion_prob: float = 0.0
    density_dropout_prob: float = 0.0
    remove_ratio: Tuple[float, float] = (0.1, 0.3)
    density_dropout_ratio: Tuple[float, float] = (0.05, 0.2)


class PointViewTransform:
    def __init__(self, cfg: Dict):
        self.cfg = PointViewConfig(**cfg)

    def _maybe_remove(self, points: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        pts = points
        if random.random() < cfg.plane_crop_prob:
            pts = plane_crop_np(pts, np.random.uniform(*cfg.remove_ratio))
        if random.random() < cfg.sphere_crop_prob:
            pts = sphere_crop_np(pts, np.random.uniform(*cfg.remove_ratio))
        if random.random() < cfg.view_occlusion_prob:
            pts = view_occlusion_np(pts, np.random.uniform(*cfg.remove_ratio))
        if random.random() < cfg.density_dropout_prob:
            pts = density_dropout_np(pts, np.random.uniform(*cfg.density_dropout_ratio))
        return pts

    def __call__(self, points: np.ndarray) -> torch.Tensor:
        cfg = self.cfg
        pts = points.astype(np.float32, copy=True)
        pts = pc_normalize_np(pts)
        if cfg.sample == "fps":
            pts = farthest_point_sample_np(pts, cfg.num_points)
        elif cfg.sample == "random":
            pts = random_resample_np(pts, cfg.num_points)
        else:
            raise ValueError(f"Unknown sample mode: {cfg.sample}")

        pts = self._maybe_remove(pts)
        pts = random_resample_np(pts, cfg.num_points)
        pts = pc_normalize_np(pts)
        if cfg.rotate:
            pts = rotate_y_np(pts)
        pts = scale_np(pts, cfg.scale[0], cfg.scale[1])
        pts = translate_np(pts, cfg.translate)
        if cfg.jitter_std > 0:
            pts = jitter_np(pts, cfg.jitter_std, cfg.jitter_clip)
        return torch.from_numpy(pts.astype(np.float32))


# --- Image transforms ---

def foreground_bbox(img: Image.Image, threshold: int = 245) -> Optional[Tuple[int, int, int, int]]:
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(arr < threshold, axis=2)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class ObjectAwareCrop:
    def __init__(self, output_size: int = 224, scale=(0.45, 0.9), threshold: int = 245):
        self.output_size = output_size
        self.scale = scale
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        bbox = foreground_bbox(img, self.threshold)
        if bbox is None:
            return T.functional.resize(img, [self.output_size, self.output_size])
        x0, y0, x1, y1 = bbox
        bw, bh = x1 - x0, y1 - y0
        crop_scale = random.uniform(*self.scale)
        cw = max(8, int(bw * crop_scale))
        ch = max(8, int(bh * crop_scale))
        cx = random.randint(x0, max(x0, x1 - cw)) if x1 - cw > x0 else max(0, x0)
        cy = random.randint(y0, max(y0, y1 - ch)) if y1 - ch > y0 else max(0, y0)
        cx = min(max(0, cx), max(0, w - cw))
        cy = min(max(0, cy), max(0, h - ch))
        cropped = img.crop((cx, cy, min(w, cx + cw), min(h, cy + ch)))
        return T.functional.resize(cropped, [self.output_size, self.output_size])


def build_image_transform(kind: str, output_size: int = 224):
    """Build image transform for MVMS-v2 views.

    Supported kinds:
      weak_full: full image resize, very mild augmentation
      full:      full image resize, standard augmentation
      crop:      object-aware crop, medium scale
      hard:      object-aware crop, aggressive scale
    """
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if kind == "weak_full":
        # Weak augmentation: just resize + minimal color jitter
        return T.Compose([
            T.Resize((output_size, output_size)),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    elif kind == "full":
        prefix = [T.Resize((output_size, output_size))]
    elif kind == "crop":
        prefix = [ObjectAwareCrop(output_size=output_size, scale=(0.55, 0.95))]
    elif kind == "hard":
        prefix = [ObjectAwareCrop(output_size=output_size, scale=(0.25, 0.55))]
    else:
        raise ValueError(f"Unknown image transform kind: {kind}")

    return T.Compose(prefix + [
        T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
