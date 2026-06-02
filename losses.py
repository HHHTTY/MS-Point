

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def nt_xent_pair(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.08) -> torch.Tensor:
    """Symmetric NT-Xent loss for two batches of embeddings."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def _add_loss(logs: Dict[str, float], name: str, value: torch.Tensor, weight: float):
    """Accumulate weighted loss and log metrics."""
    weighted = value * float(weight)
    logs[name] = float(value.detach().cpu())
    logs[f"{name}_weight"] = float(weight)
    logs[f"{name}_weighted"] = float(weighted.detach().cpu())
    return weighted


def _cosine_interp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * (1 - math.cos(progress * math.pi)) / 2


def _get_smooth_weights(epoch: int, schedule_cfg: Dict) -> Dict[str, float]:
    """Smooth continuous weight scheduling using cosine interpolation.

    Config format:
      loss_schedule:
        smooth: true
        total_epochs: 600
        weights:
          clean_cross:
            start: 1.30
            end: 0.85
          local:
            ramp_start: 80
            ramp_end: 280
            start_weight: 0.03
            end_weight: 0.20
    """
    total = schedule_cfg.get("total_epochs", 600)
    progress = min(epoch / max(1, total), 1.0)

    weights = {}
    weight_cfgs = schedule_cfg.get("weights", {})

    for name, cfg in weight_cfgs.items():
        if name == "local":
            # Special handling for local: disabled until ramp_start
            ramp_start = int(cfg.get("ramp_start", 80))
            ramp_end = int(cfg.get("ramp_end", 280))
            ramp_lo = float(cfg.get("start_weight", 0.03))
            ramp_hi = float(cfg.get("end_weight", 0.20))

            if epoch < ramp_start:
                weights[name] = 0.0
            elif epoch < ramp_end:
                p = (epoch - ramp_start) / max(1, ramp_end - ramp_start)
                weights[name] = _cosine_interp(ramp_lo, ramp_hi, p)
            else:
                weights[name] = ramp_hi
        else:
            start_w = float(cfg.get("start", 1.0))
            end_w = float(cfg.get("end", 1.0))
            weights[name] = _cosine_interp(start_w, end_w, progress)

    return weights


def mvms_v2_improved_loss(
    outputs: Dict,
    cfg: Dict,
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    temperature = float(cfg.get("temperature", 0.08))
    schedule_cfg = cfg.get("loss_schedule", {})
    dynamic_w = _get_smooth_weights(epoch, schedule_cfg)

    point = {k: v["proj"] for k, v in outputs["points"].items()}
    image = outputs["images"]
    device = next(iter(image.values())).device

    total = torch.tensor(0.0, device=device)
    logs = {"epoch": epoch}
    for k, v in dynamic_w.items():
        logs[f"dyn_weight_{k}"] = float(v)

    # 1. Clean cross-modal: clean_global <-> weak_full
    #    Core clean semantic anchor for ModelNet40 performance
    w = dynamic_w.get("clean_cross", 1.0)
    if w > 0 and "clean_global" in point and "weak_full" in image:
        loss = nt_xent_pair(point["clean_global"], image["weak_full"], temperature)
        total = total + _add_loss(logs, "loss_clean_cross", loss, w)

    # 2. Clean consistency: clean_global <-> global
    #    Bridges clean and robust global representations
    w = dynamic_w.get("clean_cons", 0.5)
    if w > 0 and "clean_global" in point and "global" in point:
        loss = nt_xent_pair(point["clean_global"], point["global"], temperature)
        total = total + _add_loss(logs, "loss_clean_cons", loss, w)

    # 3. Point scale chain: ENHANCED with clean_global bridge
    #    Now includes clean_global <-> global pair for semantic bridge
    w = dynamic_w.get("point_scale", 0.5)
    if w > 0:
        pairs = []
        if "clean_global" in point and "global" in point:
            pairs.append(("clean_global", "global", 0.5))
        if "global" in point and "mid" in point:
            pairs.append(("global", "mid", 1.0))
        if "mid" in point and "local" in point:
            pairs.append(("mid", "local", 1.0))
        if "global" in point and "local" in point:
            pairs.append(("global", "local", 0.5))
        if pairs:
            loss = sum(
                nt_xent_pair(point[a], point[b], temperature) * pw
                for a, b, pw in pairs
            ) / sum(pw for _, _, pw in pairs)
            total = total + _add_loss(logs, "loss_point_scale", loss, w)

    # 4. Image view chain: ENHANCED with weak_full <-> crop bridge
    #    Connects clean image features to mid-level robust features
    w = dynamic_w.get("image_view", 0.3)
    if w > 0:
        pairs = []
        if "weak_full" in image and "full" in image:
            pairs.append(("weak_full", "full", 1.0))
        if "full" in image and "crop" in image:
            pairs.append(("full", "crop", 1.0))
        if "crop" in image and "hard" in image:
            pairs.append(("crop", "hard", 1.0))
        if "weak_full" in image and "crop" in image:
            pairs.append(("weak_full", "crop", 0.5))
        if pairs:
            loss = sum(
                nt_xent_pair(image[a], image[b], temperature) * pw
                for a, b, pw in pairs
            ) / sum(pw for _, _, pw in pairs)
            total = total + _add_loss(logs, "loss_image_view", loss, w)

    # 5. Robust cross-modal pairs (configurable)
    w = dynamic_w.get("robust_cross", 0.8)
    if w > 0:
        cross_pairs = cfg.get("robust_cross_pairs", [
            ["global", "full", 1.0],
            ["mid", "crop", 0.7],
            ["global", "crop", 0.2],
            ["mid", "full", 0.2],
        ])
        cross_loss = torch.tensor(0.0, device=device)
        cross_wsum = 0.0
        for p_name, i_name, pair_w in cross_pairs:
            if p_name in point and i_name in image and pair_w > 0:
                cross_loss = cross_loss + nt_xent_pair(
                    point[p_name], image[i_name], temperature
                ) * float(pair_w)
                cross_wsum += float(pair_w)
        if cross_wsum > 0:
            cross_loss = cross_loss / cross_wsum
            total = total + _add_loss(logs, "loss_robust_cross", cross_loss, w)

    # 6. Local cross-modal: local <-> hard (curriculum weighted)
    w = dynamic_w.get("local", 0.0)
    if w > 0 and "local" in point and "hard" in image:
        loss = nt_xent_pair(point["local"], image["hard"], temperature)
        total = total + _add_loss(logs, "loss_local", loss, w)

    logs["loss_total"] = float(total.detach().cpu())
    return total, logs
