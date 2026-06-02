"""CR-MVMS-B-GeoMAE-Adaptive Loss Functions.

Modified from CR-MVMS-B-GeoMAE losses with adaptive weight support.

Key difference from CR-MVMS-B-GeoMAE:
  - geo_mae_weight_override parameter: when provided, overrides the scheduled weight
  - This allows the AdaptiveGeoMAEWeight controller to dynamically set the weight
  - All other loss components remain identical to CR-MVMS-B-GeoMAE

Loss components (7, same as CR-MVMS-B-GeoMAE):
  1. clean_cross:   clean_global <-> weak_full (clean semantic anchor)
  2. clean_cons:    clean_global <-> global (clean-robust consistency)
  3. point_scale:   Multi-scale point chain with semantic bridge
  4. image_view:    Multi-scale image chain with semantic bridge
  5. robust_cross:  Weighted cross-modal pairs
  6. local:         local <-> hard (curriculum weighted)
  7. geo_mae:       Patch center + local xyz reconstruction (ADAPTIVE WEIGHT)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def nt_xent_pair(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
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
    """Cosine interpolation: slow start, fast middle, slow end."""
    return start + (end - start) * (1 - math.cos(progress * math.pi)) / 2


def _get_smooth_weights(epoch: int, schedule_cfg: Dict) -> Dict[str, float]:
    """Smooth continuous weight scheduling using cosine interpolation.

    Same as CR-MVMS-B-GeoMAE (and MVMS-v2 Improved B).
    """
    total = schedule_cfg.get("total_epochs", 600)
    progress = min(epoch / max(1, total), 1.0)

    weights = {}
    weight_cfgs = schedule_cfg.get("weights", {})

    for name, cfg in weight_cfgs.items():
        if name in ("local", "geo_mae"):
            ramp_start = int(cfg.get("ramp_start", 0))
            ramp_end = int(cfg.get("ramp_end", 100))
            ramp_lo = float(cfg.get("start_weight", 0.0))
            ramp_hi = float(cfg.get("end_weight", 0.5))

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


def _compute_geo_mae_loss(
    pred_center: torch.Tensor,
    target_center: torch.Tensor,
    pred_local_xyz: torch.Tensor,
    target_local_xyz: torch.Tensor,
    mask: torch.Tensor,
    beta_xyz: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute GeoMAE reconstruction loss (same as CR-MVMS-B-GeoMAE)."""
    mask_bool = mask.bool()
    n_masked = mask_bool.sum(dim=1).clamp(min=1).float()

    # Center loss: SmoothL1
    center_diff = pred_center - target_center
    center_loss_per_patch = F.smooth_l1_loss(
        center_diff, torch.zeros_like(center_diff), reduction='none'
    ).mean(dim=-1)
    loss_center = (center_loss_per_patch * mask_bool.float()).sum(dim=1) / n_masked
    loss_center = loss_center.mean()

    # Local xyz loss: SmoothL1
    local_diff = pred_local_xyz - target_local_xyz
    local_loss_per_point = F.smooth_l1_loss(
        local_diff, torch.zeros_like(local_diff), reduction='none'
    ).mean(dim=-1)
    local_loss_per_patch = local_loss_per_point.mean(dim=-1)
    loss_local_xyz = (local_loss_per_patch * mask_bool.float()).sum(dim=1) / n_masked
    loss_local_xyz = loss_local_xyz.mean()

    # Total
    loss_total = loss_center + beta_xyz * loss_local_xyz

    logs = {
        "loss_geo_mae_center": float(loss_center.detach().cpu()),
        "loss_geo_mae_local_xyz": float(loss_local_xyz.detach().cpu()),
        "loss_geo_mae_raw": float(loss_total.detach().cpu()),
    }

    return loss_total, logs


def cr_mvms_b_geomae_adaptive_loss(
    outputs: Dict,
    cfg: Dict,
    epoch: int = 0,
    geo_mae_weight_override: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """CR-MVMS-B-GeoMAE-Adaptive combined loss.

    Identical to cr_mvms_b_geomae_loss except:
      - geo_mae_weight_override: if provided, overrides the scheduled weight
        for the GeoMAE reconstruction loss. This allows the adaptive controller
        to dynamically set the weight.

    Args:
        outputs: Model outputs (same format as CRMVMSBGeoMAE)
        cfg: Training config dict
        epoch: Current epoch
        geo_mae_weight_override: If not None, use this weight for geo_mae
                                 instead of the scheduled weight
    """
    temperature = float(cfg.get("temperature", 0.07))
    schedule_cfg = cfg.get("loss_schedule", {})
    dynamic_w = _get_smooth_weights(epoch, schedule_cfg)

    point = {k: v["proj"] for k, v in outputs["points"].items()}
    image = outputs["images"]
    device = next(iter(image.values())).device

    total = torch.tensor(0.0, device=device)
    logs = {"epoch": epoch}
    for k, v in dynamic_w.items():
        logs[f"dyn_weight_{k}"] = float(v)

    # ---- CONTRASTIVE LOSSES (identical to CR-MVMS-B-GeoMAE) ----

    # 1. Clean cross-modal: clean_global <-> weak_full
    w = dynamic_w.get("clean_cross", 1.0)
    if w > 0 and "clean_global" in point and "weak_full" in image:
        loss = nt_xent_pair(point["clean_global"], image["weak_full"], temperature)
        total = total + _add_loss(logs, "loss_clean_cross", loss, w)

    # 2. Clean consistency: clean_global <-> global
    w = dynamic_w.get("clean_cons", 0.5)
    if w > 0 and "clean_global" in point and "global" in point:
        loss = nt_xent_pair(point["clean_global"], point["global"], temperature)
        total = total + _add_loss(logs, "loss_clean_cons", loss, w)

    # 3. Point scale chain (enhanced with clean_global bridge)
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

    # 4. Image view chain (enhanced with weak_full <-> crop bridge)
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

    # 5. Robust cross-modal pairs
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

    # ---- GEOMAE RECONSTRUCTION LOSS (ADAPTIVE WEIGHT) ----

    # Determine weight: use override if provided, otherwise use schedule
    if geo_mae_weight_override is not None:
        w_recon = float(geo_mae_weight_override)
        logs["geo_mae_weight_source"] = "adaptive"
    else:
        w_recon = dynamic_w.get("geo_mae", 0.0)
        logs["geo_mae_weight_source"] = "schedule"

    if w_recon > 0 and "mae" in outputs and len(outputs["mae"]) > 0:
        mae = outputs["mae"]
        beta_xyz = float(cfg.get("geo_mae_beta_xyz", 1.0))

        loss_recon_raw, recon_logs = _compute_geo_mae_loss(
            pred_center=mae["pred_center"],
            target_center=mae["target_center"],
            pred_local_xyz=mae["pred_local_xyz"],
            target_local_xyz=mae["target_local_xyz"],
            mask=mae["mask"],
            beta_xyz=beta_xyz,
        )

        # Add reconstruction logs
        for k, v in recon_logs.items():
            logs[k] = v

        # Weight and add to total
        loss_recon_weighted = loss_recon_raw * w_recon
        logs["loss_reconstruction_weighted"] = float(loss_recon_weighted.detach().cpu())
        logs["loss_reconstruction_weight"] = float(w_recon)
        total = total + loss_recon_weighted

    # Store contrastive total for ratio computation
    recon_weighted = logs.get("loss_reconstruction_weighted", 0.0)
    logs["loss_contrast_total"] = float(total.detach().cpu()) - recon_weighted

    # Compute loss ratio for adaptive controller
    contrast_total = logs["loss_contrast_total"]
    if contrast_total > 1e-8 and recon_weighted > 0:
        logs["loss_ratio_recon"] = recon_weighted / contrast_total
    else:
        logs["loss_ratio_recon"] = 0.0

    logs["loss_total"] = float(total.detach().cpu())
    return total, logs
