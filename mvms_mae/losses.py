"""MVMS-MAE losses: MVMS-v2 contrastive losses + MAE reconstruction loss.

Loss components:
  1. clean_cross:   NTXent(point_clean_global, image_weak_full)
  2. clean_cons:    NTXent(point_clean_global, point_global)
  3. point_scale:   NTXent across point scales
  4. image_view:    NTXent across image views
  5. robust_cross:  NTXent point-image cross pairs
  6. local:         NTXent(point_local, image_hard) with curriculum
  7. reconstruction: Chamfer distance on masked patches (MAE)

Total: L = L_mvms + lambda_rec * alpha(epoch) * L_rec
  alpha warmup: epoch 0-20 -> 0, 20-100 -> linear 0->1, 100+ -> 1
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def nt_xent_pair(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
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


def _get_dynamic_weights(epoch: int, schedule: Dict) -> Dict[str, float]:
    """Compute dynamic loss weights based on epoch and schedule config."""
    phases = schedule.get("phases", [])
    if not phases:
        return schedule.get("weights", {})

    current_weights = {}
    for i, phase in enumerate(phases):
        if epoch < phase["end_epoch"]:
            current_weights = dict(phase["weights"])
            break
    else:
        current_weights = dict(phases[-1]["weights"])

    local_ramp = schedule.get("local_ramp", {})
    if local_ramp:
        ramp_start = int(local_ramp.get("start_epoch", 100))
        ramp_end = int(local_ramp.get("end_epoch", 300))
        ramp_lo = float(local_ramp.get("start_weight", 0.05))
        ramp_hi = float(local_ramp.get("end_weight", 0.15))
        if epoch < ramp_start:
            current_weights["local"] = 0.0
        elif epoch >= ramp_end:
            current_weights["local"] = ramp_hi
        else:
            alpha = float(epoch - ramp_start) / max(1, ramp_end - ramp_start)
            current_weights["local"] = ramp_lo + alpha * (ramp_hi - ramp_lo)

    return current_weights


def chamfer_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Chamfer distance (L2) between two point sets.
    pred: [M, S, 3], target: [M, S, 3]
    """
    # pred/target: [M, S, 3]
    dist = torch.cdist(pred, target, p=2) ** 2  # [M, S, S]
    min_dim2 = dist.min(dim=2).values.mean(dim=1)  # [M]
    min_dim1 = dist.min(dim=1).values.mean(dim=1)  # [M]
    return (min_dim2 + min_dim1).mean()


def mae_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MAE reconstruction loss using Chamfer distance on masked patches only.

    Args:
        pred: [B, G, S, 3] - predicted local_xyz for all patches
        target: [B, G, S, 3] - target local_xyz for all patches
        mask: [B, G] - 1=masked, 0=visible

    Returns:
        scalar loss (Chamfer distance averaged over batch)
    """
    B, G, S, C = pred.shape
    total_loss = torch.tensor(0.0, device=pred.device)
    count = 0

    for b in range(B):
        mask_b = mask[b].bool()  # [G]
        if mask_b.sum() == 0:
            continue
        pred_masked = pred[b, mask_b]    # [M, S, 3]
        target_masked = target[b, mask_b]  # [M, S, 3]
        total_loss = total_loss + chamfer_l2(pred_masked, target_masked)
        count += 1

    if count == 0:
        return pred.sum() * 0.0
    return total_loss / count


def mae_alpha_schedule(epoch: int, start_epoch: int = 20, end_epoch: int = 100) -> float:
    """Warmup schedule for MAE loss weight.

    epoch 0 to start_epoch:   alpha = 0 (MAE disabled)
    start_epoch to end_epoch:  alpha linearly 0 -> 1
    end_epoch+:               alpha = 1
    """
    if epoch < start_epoch:
        return 0.0
    elif epoch < end_epoch:
        return float(epoch - start_epoch) / float(end_epoch - start_epoch)
    else:
        return 1.0


def mvms_mae_loss(
    outputs: Dict,
    cfg: Dict,
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """MVMS-MAE combined loss.

    L_total = L_mvms + lambda_rec * alpha(epoch) * L_rec

    Args:
        outputs: model forward pass output
        cfg: training config dict
        epoch: current epoch number
    """
    temperature = float(cfg.get("temperature", 0.1))
    schedule_cfg = cfg.get("loss_schedule", {})
    dynamic_w = _get_dynamic_weights(epoch, schedule_cfg)

    point = {k: v["proj"] for k, v in outputs["points"].items()}
    image = outputs["images"]
    device = next(iter(image.values())).device

    total = torch.tensor(0.0, device=device)
    logs = {"epoch": epoch}
    for k, v in dynamic_w.items():
        logs[f"dyn_weight_{k}"] = float(v)

    # ============ MVMS-v2 contrastive losses ============

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

    # 3. Point scale chain
    w = dynamic_w.get("point_scale", 0.5)
    if w > 0:
        pairs = []
        if "global" in point and "mid" in point:
            pairs.append(("global", "mid"))
        if "mid" in point and "local" in point:
            pairs.append(("mid", "local"))
        if "global" in point and "local" in point:
            pairs.append(("global", "local"))
        if pairs:
            loss = sum(nt_xent_pair(point[a], point[b], temperature) for a, b in pairs) / len(pairs)
            total = total + _add_loss(logs, "loss_point_scale", loss, w)

    # 4. Image view chain
    w = dynamic_w.get("image_view", 0.3)
    if w > 0:
        pairs = []
        if "weak_full" in image and "full" in image:
            pairs.append(("weak_full", "full"))
        if "full" in image and "crop" in image:
            pairs.append(("full", "crop"))
        if "crop" in image and "hard" in image:
            pairs.append(("crop", "hard"))
        if pairs:
            loss = sum(nt_xent_pair(image[a], image[b], temperature) for a, b in pairs) / len(pairs)
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
                cross_loss = cross_loss + nt_xent_pair(point[p_name], image[i_name], temperature) * float(pair_w)
                cross_wsum += float(pair_w)
        if cross_wsum > 0:
            cross_loss = cross_loss / cross_wsum
            total = total + _add_loss(logs, "loss_robust_cross", cross_loss, w)

    # 6. Local cross-modal: local <-> hard
    w = dynamic_w.get("local", 0.0)
    if w > 0 and "local" in point and "hard" in image:
        loss = nt_xent_pair(point["local"], image["hard"], temperature)
        total = total + _add_loss(logs, "loss_local", loss, w)

    # ============ MAE reconstruction loss ============

    mae_schedule_cfg = cfg.get("mae_schedule", {})
    rec_lambda = float(cfg.get("loss_weights", {}).get("reconstruction", 0.10))
    alpha = mae_alpha_schedule(
        epoch,
        start_epoch=int(mae_schedule_cfg.get("start_epoch", 20)),
        end_epoch=int(mae_schedule_cfg.get("end_epoch", 100)),
    )
    logs["mae_alpha"] = alpha
    logs["recon_lambda"] = rec_lambda

    if "mae" in outputs and outputs["mae"] is not None:
        mae_out = outputs["mae"]
        pred = mae_out["pred"]       # [B, G, S, 3]
        target = mae_out["target"]   # [B, G, S, 3]
        mask = mae_out["mask"]       # [B, G]

        rec_loss_raw = mae_reconstruction_loss(pred, target, mask)
        rec_loss_weighted = rec_lambda * alpha * rec_loss_raw

        logs["loss_reconstruction_raw"] = float(rec_loss_raw.detach().cpu())
        logs["loss_reconstruction_weighted"] = float(rec_loss_weighted.detach().cpu())

        # Log gradient norms separately for monitoring
        if rec_loss_raw.requires_grad:
            logs["mask_ratio_actual"] = float(mask.mean().detach().cpu())

        total = total + rec_loss_weighted
    else:
        logs["loss_reconstruction_raw"] = 0.0
        logs["loss_reconstruction_weighted"] = 0.0
        logs["mask_ratio_actual"] = 0.0

    logs["loss_total"] = float(total.detach().cpu())
    return total, logs
