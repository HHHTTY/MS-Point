
import glob
import os
import random
from typing import Dict, List

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from datasets.plyfile import load_ply
from .augmentations import PointViewTransform, build_image_transform

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


class ShapeNetMVMSv2(Dataset):
    """MVMS-v2 dataset with clean/robust views and same_base_image option."""

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pcd_path = self.data[idx]
        point_cloud = load_ply(pcd_path)
        render_paths = get_render_imgs(pcd_path)
        if len(render_paths) < 1:
            raise RuntimeError(f"No render images for {pcd_path}")

        # Apply point transforms (each view independently)
        points = {
            name: transform(point_cloud.copy())
            for name, transform in self.point_transforms.items()
        }

        # Apply image transforms
        images = {}
        if self.same_base_image:
            # All image views from the same base render for semantic consistency
            img_path = random.choice(render_paths)
            base_img = Image.open(img_path).convert("RGB")
            for name, transform in self.image_transforms.items():
                images[name] = transform(base_img.copy())
        else:
            # Each image view from potentially different renders (original MVMS behavior)
            n_views = len(self.image_transforms)
            chosen = random.sample(render_paths, min(n_views, len(render_paths)))
            while len(chosen) < n_views:
                chosen.append(random.choice(render_paths))
            for (name, transform), img_path in zip(self.image_transforms.items(), chosen):
                images[name] = transform(Image.open(img_path).convert("RGB"))

        return {"points": points, "images": images, "path": pcd_path}
