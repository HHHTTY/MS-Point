
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from datasets.data1 import ModelNet40SVM, ScanObjectNNSVM
from teacher_guided_geomae_experiments import (
    TeacherGuidedGeoMAE,
    teacher_guided_geomae_loss,
    ShapeNetCRGeoMAE,
    GradMonitor,
    AdaptiveGeoMAEWeight,
)


# ─── Utilities ────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def snapshot_sources(run_dir, config_path, script_path, module_dir):
    """Snapshot config, training script, and experiment module into run dir."""
    shutil.copy2(config_path, os.path.join(run_dir, "config.yaml"))
    shutil.copy2(script_path, os.path.join(run_dir, os.path.basename(script_path) + ".bak"))
    dst = os.path.join(run_dir, os.path.basename(module_dir) + "_snapshot")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(module_dir, dst)


# ─── LR Scheduler with Warmup ────────────────────────────────────────────────

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-5):
    """Combined linear warmup + cosine annealing LR scheduler."""
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    eta_min_ratio = eta_min / base_lrs[0] if base_lrs[0] > 0 else 0

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Feature Extraction & Evaluation ─────────────────────────────────────────

def extract_features(model, loader, device, feature_name="feat"):
    """Extract features from model on a dataset."""
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.eval()
    features, labels = [], []
    with torch.no_grad():
        for points, target in loader:
            result = raw_model.encode_point_tensor(points.to(device), "global")
            features.append(result[feature_name].cpu().numpy())
            labels.append(target.numpy())
    raw_model.train()
    return np.concatenate(features), np.concatenate(labels)


def linear_eval(model, cfg, run_dir, device, epoch, eval_log_path, summary_log_path):
    """Run Linear SVM evaluation on ModelNet40 and ScanObjectNN."""
    from sklearn.svm import SVC

    eval_cfg = cfg.get("eval", {})
    datasets_cfg = eval_cfg.get("datasets", ["modelnet40", "scanobjectnn"])
    features_to_eval = eval_cfg.get("features", ["feat", "proj"])
    c_values = eval_cfg.get("c_values", [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1])
    scan_threshold = float(eval_cfg.get("scan_threshold", 0.878))
    balanced_weights = eval_cfg.get("balanced_weights", {"modelnet40": 0.55, "scanobjectnn": 0.45})

    results = []
    all_best = {}

    for ds_name in datasets_cfg:
        ds_key = ds_name.lower()

        if ds_key == "modelnet40":
            train_loader = DataLoader(
                ModelNet40SVM(num_points=cfg["model"]["num_points"], partition='train'),
                batch_size=128, shuffle=False, num_workers=4)
            test_loader = DataLoader(
                ModelNet40SVM(num_points=cfg["model"]["num_points"], partition='test'),
                batch_size=128, shuffle=False, num_workers=4)
        elif ds_key == "scanobjectnn":
            train_loader = DataLoader(
                ScanObjectNNSVM(num_points=cfg["model"]["num_points"], partition='train'),
                batch_size=128, shuffle=False, num_workers=4)
            test_loader = DataLoader(
                ScanObjectNNSVM(num_points=cfg["model"]["num_points"], partition='test'),
                batch_size=128, shuffle=False, num_workers=4)
        else:
            continue

        for feat_name in features_to_eval:
            try:
                train_feats, train_labels = extract_features(model, train_loader, device, feat_name)
                test_feats, test_labels = extract_features(model, test_loader, device, feat_name)
            except Exception as e:
                print(f"  [Eval] Error extracting {feat_name} from {ds_name}: {e}")
                continue

            best_acc, best_c = 0, None
            for c in c_values:
                try:
                    svm = SVC(C=float(c), kernel="linear", cache_size=4096)
                    svm.fit(train_feats, train_labels)
                    acc = float(svm.score(test_feats, test_labels))
                    if acc > best_acc:
                        best_acc, best_c = acc, c
                except Exception:
                    continue

            result = {
                "epoch": epoch, "dataset": ds_name, "feature": feat_name,
                "best_acc": round(best_acc, 5), "best_c": best_c,
            }
            results.append(result)
            append_jsonl(eval_log_path, result)
            key = f"{ds_key}_{feat_name}"
            all_best[key] = best_acc
            print(f"  {ds_name:15s} {feat_name:5s}: {best_acc:.4f} (C={best_c})")

    csv_path = os.path.join(run_dir, f"eval_epoch_{epoch}.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)

    mn_acc = all_best.get("modelnet40_feat", 0)
    sn_acc = all_best.get("scanobjectnn_feat", 0)
    mn_w = balanced_weights.get("modelnet40", 0.55)
    sn_w = balanced_weights.get("scanobjectnn", 0.45)

    balanced = mn_w * mn_acc + sn_w * sn_acc
    if sn_acc < scan_threshold:
        balanced -= 2.0 * (scan_threshold - sn_acc)

    summary = {
        "epoch": epoch,
        "modelnet40_feat": mn_acc,
        "scanobjectnn_feat": sn_acc,
        "balanced_score": round(balanced, 5),
    }
    append_jsonl(summary_log_path, summary)
    print(f"  Balanced score: {balanced:.4f} (MN={mn_acc:.4f}, SN={sn_acc:.4f})")

    return summary


def save_checkpoint(state, path):
    torch.save(state, path)


# ─── Main Training Loop ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Teacher-Guided GeoMAE Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry_run_batches", type=int, default=0)
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_name = cfg.get("name", "teacher_guided_geomae")
    seed = cfg.get("seed", 2024)
    set_seed(seed)

    # Create run directory
    run_root = cfg.get("run_root", "/data/mn/yht_pd/cjs_experiments/teacher_guided_geomae_runs")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(run_root, f"{cfg_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Snapshot code
    snapshot_sources(
        run_dir,
        args.config,
        os.path.abspath(__file__),
        os.path.join(os.path.dirname(__file__), "teacher_guided_geomae_experiments"),
    )
    append_jsonl(os.path.join(run_dir, "run_meta.jsonl"), {
        "name": cfg_name, "timestamp": timestamp, "config": args.config,
        "resume": args.resume, "experiment_type": "teacher_guided_geomae",
    })

    # Paths
    train_log = os.path.join(run_dir, "train_metrics.jsonl")
    eval_log = os.path.join(run_dir, "eval_metrics.jsonl")
    summary_log = os.path.join(run_dir, "eval_summary.jsonl")
    grad_log = os.path.join(run_dir, "grad_metrics.jsonl")
    adaptive_log = os.path.join(run_dir, "adaptive_weight.jsonl")
    kd_log = os.path.join(run_dir, "kd_metrics.jsonl")

    # Training params
    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 600)
    batch_size = train_cfg.get("batch_size", 20)
    lr = train_cfg.get("lr", 0.0010)
    weight_decay = train_cfg.get("weight_decay", 1e-5)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    grad_clip = train_cfg.get("grad_clip", 10.0)
    eval_every = train_cfg.get("eval_every", 20)
    save_every = train_cfg.get("save_every", 100)
    print_freq = train_cfg.get("print_freq", 100)
    num_workers = train_cfg.get("num_workers", 6)
    grad_accum_steps = train_cfg.get("grad_accum_steps", 3)

    # GradMonitor config
    grad_mon_cfg = train_cfg.get("grad_monitor", {})
    grad_mon_enabled = bool(grad_mon_cfg.get("enabled", True))
    grad_mon_interval = int(grad_mon_cfg.get("log_interval", 100))

    # Adaptive weight config
    adaptive_cfg = train_cfg.get("adaptive_weight", {})
    adaptive_enabled = bool(adaptive_cfg.get("enabled", True))

    # GPU setup
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", True)

    # Dataset
    print("Loading dataset...")
    dataset = ShapeNetCRGeoMAE(cfg.get("data", {}))
    print(f"Dataset size: {len(dataset)}")
    effective_bs = batch_size * grad_accum_steps
    print(f"Batch size: {batch_size}, grad_accum: {grad_accum_steps}, effective: {effective_bs}")

    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    # Model
    model_cfg = cfg.get("model", {})
    model_args = argparse.Namespace(
        emb_dims=model_cfg.get("emb_dims", 1024),
        k=model_cfg.get("k", 15),
        dropout=model_cfg.get("dropout", 0.5),
    )
    model = TeacherGuidedGeoMAE(model_args, model_cfg)

    # Load teacher (frozen)
    teacher_cfg = cfg.get("teacher", {})
    teacher_ckpt_path = teacher_cfg.get("checkpoint", "")
    if teacher_ckpt_path and os.path.isfile(teacher_ckpt_path):
        model.load_teacher(teacher_ckpt_path)
    else:
        print(f"WARNING: Teacher checkpoint not found at {teacher_ckpt_path}")
        print("Training will proceed without teacher guidance!")

    # Load student initialization (warm start)
    student_cfg = cfg.get("student_init", {})
    student_ckpt_path = student_cfg.get("checkpoint", "")
    if student_ckpt_path and os.path.isfile(student_ckpt_path):
        model.load_student(student_ckpt_path)
    else:
        print(f"WARNING: Student init checkpoint not found at {student_ckpt_path}")
        print("Student will train from scratch!")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    teacher_params = sum(p.numel() for p in model.teacher_point_encoder.parameters()) + \
                     sum(p.numel() for p in model.teacher_point_projectors.parameters())
    mae_params = sum(p.numel() for p in model.mae_decoder.parameters()) if model.mae_enabled else 0
    student_base_params = total_params - teacher_params - mae_params

    print(f"\nParameter counts:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Teacher params (frozen): {teacher_params:,}")
    print(f"  Student base params: {student_base_params:,}")
    print(f"  GeoMAE decoder params: {mae_params:,}")

    model = model.to(device)

    # Ensure teacher is in eval mode on device
    model.teacher_point_encoder.eval()
    model.teacher_point_projectors.eval()

    # GradMonitor
    grad_monitor = None
    if grad_mon_enabled:
        grad_monitor = GradMonitor(
            target_ratio=float(grad_mon_cfg.get("target_ratio", 0.10)),
            auto_tune=False,
            tune_interval=int(grad_mon_cfg.get("tune_interval", 200)),
            tune_patience=int(grad_mon_cfg.get("tune_patience", 5)),
        )

    # Adaptive weight controller
    adaptive_controller = None
    if adaptive_enabled:
        schedule_cfg = train_cfg.get("loss_schedule", {}).get("weights", {}).get("geo_mae", {})
        adaptive_controller = AdaptiveGeoMAEWeight(
            target_ratio=float(adaptive_cfg.get("target_ratio", 0.10)),
            adjustment_freq=int(adaptive_cfg.get("adjustment_freq", 100)),
            lr=float(adaptive_cfg.get("lr", 0.3)),
            w_min=float(adaptive_cfg.get("w_min", 0.10)),
            w_max=float(adaptive_cfg.get("w_max", 3.0)),
            ema_alpha=float(adaptive_cfg.get("ema_alpha", 0.3)),
            warmup_epochs=int(adaptive_cfg.get("warmup_epochs", 40)),
            initial_weight=float(adaptive_cfg.get("initial_weight", 0.50)),
            schedule_cfg=schedule_cfg,
        )

    # Mixed precision (AMP) for memory efficiency
    use_amp = bool(train_cfg.get("use_amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    # Scheduler
    scheduler = get_warmup_cosine_scheduler(
        optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs,
        eta_min=train_cfg.get("eta_min", 1e-5),
    )

    # Resume
    start_epoch = 0
    best_mn, best_sn, best_balanced = 0, 0, -float("inf")
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
        model.load_state_dict(state_dict, strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resumed from epoch {start_epoch}")

        if adaptive_controller and "adaptive_controller_state" in ckpt:
            adaptive_controller.load_state(ckpt["adaptive_controller_state"])
            print(f"  Restored adaptive controller: {adaptive_controller.summary()}")
        elif adaptive_controller:
            initial_w = adaptive_controller.get_weight(start_epoch)
            adaptive_controller.current_weight = initial_w
            print(f"  Adaptive controller initialized with scheduled weight: {initial_w:.4f}")

        # Rebuild optimizer and scheduler
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        scheduler = get_warmup_cosine_scheduler(
            optimizer, warmup_epochs=min(warmup_epochs, max(0, 2)),
            total_epochs=epochs, eta_min=train_cfg.get("eta_min", 1e-5),
        )
        for _ in range(start_epoch):
            scheduler.step()

        best_mn = ckpt.get("best_modelnet", 0)
        best_sn = ckpt.get("best_scan", 0)
        best_balanced = ckpt.get("best_balanced", -float("inf"))

    print(f"\n{'='*70}")
    print(f"Teacher-Guided GeoMAE Training")
    print(f"{'='*70}")
    print(f"Start epoch: {start_epoch}, Total epochs: {epochs}")
    print(f"Learning rate: {lr}, Warmup: {warmup_epochs} epochs")
    print(f"Evaluation every {eval_every} epochs, save every {save_every} epochs")
    print(f"Effective batch size: {effective_bs}")
    print(f"GradMonitor: {'enabled' if grad_mon_enabled else 'disabled'}")
    print(f"AdaptiveGeoMAEWeight: {'enabled' if adaptive_controller else 'disabled'}")
    if adaptive_controller:
        print(f"  target_ratio={adaptive_controller.target_ratio}")
        print(f"  controller_lr={adaptive_controller.lr}")
        print(f"  warmup_epochs={adaptive_controller.warmup_epochs}")
    print(f"Teacher: {'loaded' if model.teacher_loaded else 'NOT LOADED'}")
    print(f"{'='*70}\n")

    # ─── Training Loop ────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        # Ensure teacher stays in eval mode
        model.teacher_point_encoder.eval()
        model.teacher_point_projectors.eval()

        epoch_metrics = {}
        n_steps = 0
        t0 = time.time()

        # Log adaptive controller state at epoch start
        if adaptive_controller:
            epoch_adaptive_state = {
                "epoch_start": epoch,
                "adaptive_weight": adaptive_controller.current_weight,
                "adaptive_ema_ratio": adaptive_controller.ema_ratio,
                "adaptive_is_warmup": adaptive_controller.is_warmup(epoch),
                "adaptive_scheduled_w": adaptive_controller._scheduled_weight(epoch),
                "adaptive_global_step": adaptive_controller.global_step,
            }
            append_jsonl(adaptive_log, epoch_adaptive_state)

        for batch_idx, batch in enumerate(loader):
            # Move to device
            points = {k: v.to(device, non_blocking=True) for k, v in batch["points"].items()}
            images = {k: v.to(device, non_blocking=True) for k, v in batch["images"].items()}
            batch_input = {"points": points, "images": images, "path": batch["path"]}

            # Move MAE data
            if "mae" in batch and len(batch["mae"]) > 0 and batch["mae"].get("mask") is not None:
                mae_input = {k: v.to(device, non_blocking=True) for k, v in batch["mae"].items()}
                batch_input["mae"] = mae_input

            # Forward with AMP autocast
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(batch_input)

                # Get adaptive weight for this step
                geo_mae_w = None
                if adaptive_controller:
                    geo_mae_w = adaptive_controller.get_weight(epoch)

                # Loss (9-component)
                loss, logs = teacher_guided_geomae_loss(
                    outputs, train_cfg, epoch,
                    geo_mae_weight_override=geo_mae_w,
                )
                loss_scaled = loss / grad_accum_steps

            # Backward with GradScaler
            scaler.scale(loss_scaled).backward()

            # Step optimizer every grad_accum_steps
            if (batch_idx + 1) % grad_accum_steps == 0:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                n_steps += 1
                global_step += 1

                # Update adaptive controller
                if adaptive_controller:
                    current_ratio = logs.get("loss_ratio_recon", 0.0)
                    new_w, adjusted = adaptive_controller.update(
                        current_ratio, epoch, batch_idx + 1,
                    )
                    if adjusted:
                        adjustment_record = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "step_in_epoch": batch_idx + 1,
                            "old_weight": logs.get("loss_reconstruction_weight", 0),
                            "new_weight": round(new_w, 6),
                            "current_ratio": round(current_ratio, 6),
                            "ema_ratio": round(adaptive_controller.ema_ratio or 0, 6),
                            "target_ratio": adaptive_controller.target_ratio,
                        }
                        append_jsonl(adaptive_log, adjustment_record)
                        print(f"  [Adaptive] w={new_w:.4f} "
                              f"(ratio={current_ratio:.4f}, ema={adaptive_controller.ema_ratio:.4f}, "
                              f"target={adaptive_controller.target_ratio:.2f})")

                # Gradient monitoring
                if grad_monitor is not None and global_step % grad_mon_interval == 0:
                    recon_weighted = logs.get("loss_reconstruction_weighted", 0.0)
                    contrast_total = logs.get("loss_contrast_total", 0.0)
                    recon_raw = logs.get("loss_geo_mae_raw", 0.0)
                    loss_ratio = recon_weighted / (contrast_total + 1e-8)

                    grad_metrics = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss_contrast_total": round(contrast_total, 6),
                        "loss_recon_weighted": round(recon_weighted, 6),
                        "loss_recon_raw": round(recon_raw, 6),
                        "loss_ratio_recon": round(loss_ratio, 6),
                        "adaptive_weight": round(geo_mae_w, 6) if geo_mae_w else None,
                    }
                    append_jsonl(grad_log, grad_metrics)

                # Log KD metrics periodically
                if global_step % grad_mon_interval == 0:
                    kd_metrics = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "kd_cos_sim_mean": logs.get("kd_cos_sim_mean", 0),
                        "kd_global_cos_sim_mean": logs.get("kd_global_cos_sim_mean", 0),
                        "consistency_cos_sim_mean": logs.get("consistency_cos_sim_mean", 0),
                        "loss_teacher_kd_weighted": logs.get("loss_teacher_kd_weighted", 0),
                        "loss_teacher_kd_global_weighted": logs.get("loss_teacher_kd_global_weighted", 0),
                        "loss_consistency_weighted": logs.get("loss_consistency_weighted", 0),
                    }
                    append_jsonl(kd_log, kd_metrics)

            # Accumulate metrics
            for k, v in logs.items():
                if k == "epoch" or not isinstance(v, (int, float)):
                    continue
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v

            # Print
            if (batch_idx + 1) % print_freq == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                msg = (f"  Epoch {epoch} step {batch_idx + 1}: "
                       f"loss={logs.get('loss_total', 0):.4f} lr={current_lr:.6f}")
                if "loss_geo_mae_raw" in logs:
                    msg += f" geo_mae={logs['loss_geo_mae_raw']:.4f}"
                if "kd_cos_sim_mean" in logs:
                    msg += f" kd_sim={logs['kd_cos_sim_mean']:.4f}"
                if "consistency_cos_sim_mean" in logs:
                    msg += f" cons_sim={logs['consistency_cos_sim_mean']:.4f}"
                if geo_mae_w is not None:
                    msg += f" mae_w={geo_mae_w:.4f}"
                print(msg)

            # Write step-level JSONL
            if (batch_idx + 1) % print_freq == 0:
                step_log = {
                    "epoch": epoch,
                    "step": batch_idx + 1,
                    "global_step": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "adaptive_weight": geo_mae_w,
                    **logs,
                }
                append_jsonl(train_log, step_log)

            if args.dry_run_batches > 0 and batch_idx >= args.dry_run_batches:
                break

        # Epoch average
        scheduler.step()
        elapsed = time.time() - t0
        avg_metrics = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "grad_accum_steps": grad_accum_steps,
            "epoch_time_s": round(elapsed, 1),
            "adaptive_weight": adaptive_controller.current_weight if adaptive_controller else None,
            "adaptive_ema_ratio": adaptive_controller.ema_ratio if adaptive_controller else None,
        }
        for k, v in epoch_metrics.items():
            avg_metrics[k] = v / len(loader)

        append_jsonl(train_log, avg_metrics)
        msg = (f"Epoch {epoch}: loss_total={avg_metrics.get('loss_total', 0):.4f} "
               f"lr={avg_metrics['lr']:.6f} time={elapsed:.0f}s")
        if "loss_geo_mae_raw" in avg_metrics:
            msg += f" geo_mae={avg_metrics['loss_geo_mae_raw']:.4f}"
        if "kd_cos_sim_mean" in avg_metrics:
            msg += f" kd_sim={avg_metrics['kd_cos_sim_mean']:.4f}"
        if adaptive_controller:
            msg += f" adapt_w={adaptive_controller.current_weight:.4f}"
        print(msg)

        # Save checkpoint every save_every epochs
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": cfg,
                "best_modelnet": best_mn,
                "best_scan": best_sn,
                "best_balanced": best_balanced,
                "global_step": global_step,
            }
            if adaptive_controller:
                ckpt_data["adaptive_controller_state"] = adaptive_controller.get_state()
            save_checkpoint(ckpt_data, os.path.join(run_dir, f"ckpt_epoch_{epoch}.pth"))
            print(f"  Saved checkpoint: ckpt_epoch_{epoch}.pth")

        # Evaluation
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            if not args.no_eval:
                print(f"\n{'='*40} Evaluation @ Epoch {epoch} {'='*40}")
                summary = linear_eval(
                    model, cfg, run_dir, device, epoch, eval_log, summary_log,
                )
                mn_acc = summary.get("modelnet40_feat", 0)
                sn_acc = summary.get("scanobjectnn_feat", 0)
                bal_score = summary.get("balanced_score", 0)

                if mn_acc > best_mn:
                    best_mn = mn_acc
                    save_checkpoint({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "adaptive_controller_state": adaptive_controller.get_state() if adaptive_controller else None,
                    }, os.path.join(run_dir, "best_modelnet_model.pth"))
                    print(f"  *** New best ModelNet40: {best_mn:.4f}")

                if sn_acc > best_sn:
                    best_sn = sn_acc
                    save_checkpoint({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "adaptive_controller_state": adaptive_controller.get_state() if adaptive_controller else None,
                    }, os.path.join(run_dir, "best_scan_model.pth"))
                    print(f"  *** New best ScanObjectNN: {best_sn:.4f}")

                if bal_score > best_balanced:
                    best_balanced = bal_score
                    save_checkpoint({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "adaptive_controller_state": adaptive_controller.get_state() if adaptive_controller else None,
                    }, os.path.join(run_dir, "best_balanced_model.pth"))
                    print(f"  *** New best balanced: {best_balanced:.4f}")

                # Progress report
                print(f"\n  Current best scores:")
                print(f"    ModelNet40: {best_mn:.4f}")
                print(f"    ScanObjectNN: {best_sn:.4f}")
                print(f"    Balanced: {best_balanced:.4f}")
                print(f"    Target (B champion): 90.62%")
                print(f"{'='*40} End Evaluation {'='*40}\n")

        if args.dry_run_batches > 0 and epoch >= 2:
            break

    # Save final
    save_checkpoint({
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "adaptive_controller_state": adaptive_controller.get_state() if adaptive_controller else None,
    }, os.path.join(run_dir, "last_model.pth"))

    # Final summary
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"  Run dir: {run_dir}")
    print(f"  Best ModelNet40: {best_mn:.4f}")
    print(f"  Best ScanObjectNN: {best_sn:.4f}")
    print(f"  Best Balanced: {best_balanced:.4f}")

    if adaptive_controller:
        print(f"\n  Adaptive controller summary:")
        print(f"    Final weight: {adaptive_controller.current_weight:.4f}")
        print(f"    Final EMA ratio: {adaptive_controller.ema_ratio:.4f if adaptive_controller.ema_ratio else 'N/A'}")
        print(f"    Total adjustments: {len(adaptive_controller.adjustment_history)}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
