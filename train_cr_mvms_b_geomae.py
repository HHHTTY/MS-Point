

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
from cr_mvms_b_geomae_experiments import (
    CRMVMSBGeoMAE,
    cr_mvms_b_geomae_loss,
    ShapeNetCRGeoMAE,
    GradMonitor,
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
    parser = argparse.ArgumentParser(description="CR-MVMS-B-GeoMAE Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry_run_batches", type=int, default=0)
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_name = cfg.get("name", "cr_mvms_b_geomae")
    seed = cfg.get("seed", 2024)
    set_seed(seed)

    # Create run directory
    run_root = cfg.get("run_root", "/data/mn/yht_pd/cjs_experiments/cr_mvms_b_geomae_runs")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(run_root, f"{cfg_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Snapshot code
    snapshot_sources(
        run_dir,
        args.config,
        os.path.abspath(__file__),
        os.path.join(os.path.dirname(__file__), "cr_mvms_b_geomae_experiments"),
    )
    append_jsonl(os.path.join(run_dir, "run_meta.jsonl"), {
        "name": cfg_name, "timestamp": timestamp, "config": args.config,
        "resume": args.resume,
    })

    # Paths
    train_log = os.path.join(run_dir, "train_metrics.jsonl")
    eval_log = os.path.join(run_dir, "eval_metrics.jsonl")
    summary_log = os.path.join(run_dir, "eval_summary.jsonl")
    grad_log = os.path.join(run_dir, "grad_metrics.jsonl")

    # Training params
    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 600)
    batch_size = train_cfg.get("batch_size", 16)
    lr = train_cfg.get("lr", 0.0014)
    weight_decay = train_cfg.get("weight_decay", 1e-5)
    warmup_epochs = train_cfg.get("warmup_epochs", 10)
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
    model = CRMVMSBGeoMAE(model_args, model_cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mae_params = sum(p.numel() for p in model.mae_decoder.parameters()) if model.mae_enabled else 0
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    if model.mae_enabled:
        print(f"GeoMAE decoder params: {mae_params:,}")
        print(f"Base model params: {total_params - mae_params:,}")

    model = model.to(device)

    # GradMonitor
    grad_monitor = None
    if grad_mon_enabled:
        grad_monitor = GradMonitor(
            target_ratio=float(grad_mon_cfg.get("target_ratio", 0.15)),
            auto_tune=False,  # No auto-tune (avoid double forward pass)
            tune_interval=int(grad_mon_cfg.get("tune_interval", 200)),
            tune_patience=int(grad_mon_cfg.get("tune_patience", 5)),
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
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
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        scheduler = get_warmup_cosine_scheduler(
            optimizer, warmup_epochs=min(warmup_epochs, max(0, 2)),
            total_epochs=epochs, eta_min=train_cfg.get("eta_min", 1e-5),
        )
        for _ in range(start_epoch):
            scheduler.step()

    print(f"Start epoch: {start_epoch}, Total epochs: {epochs}")
    print(f"Learning rate: {lr}, Warmup: {warmup_epochs} epochs")
    print(f"Evaluation every {eval_every} epochs, save every {save_every} epochs")
    print(f"Effective batch size: {effective_bs}")
    print(f"GradMonitor: {'enabled' if grad_mon_enabled else 'disabled'} (loss-ratio mode)")
    print("=" * 60)

    # ─── Training Loop ────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_metrics = {}
        n_steps = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            # Move to device
            points = {k: v.to(device, non_blocking=True) for k, v in batch["points"].items()}
            images = {k: v.to(device, non_blocking=True) for k, v in batch["images"].items()}
            batch_input = {"points": points, "images": images, "path": batch["path"]}

            # Move MAE data
            if "mae" in batch and len(batch["mae"]) > 0 and batch["mae"].get("mask") is not None:
                mae_input = {k: v.to(device, non_blocking=True) for k, v in batch["mae"].items()}
                batch_input["mae"] = mae_input

            # Forward
            outputs = model(batch_input)

            # Loss
            loss, logs = cr_mvms_b_geomae_loss(outputs, cfg.get("training", {}), epoch)
            loss_scaled = loss / grad_accum_steps

            # Backward
            loss_scaled.backward()

            # Step optimizer every grad_accum_steps
            if (batch_idx + 1) % grad_accum_steps == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                n_steps += 1
                global_step += 1

                # Gradient monitoring via loss-value ratio (no extra forward pass)
                if grad_monitor is not None and global_step % grad_mon_interval == 0:
                    recon_weighted = logs.get("loss_reconstruction_weighted", 0.0)
                    contrast_total = logs.get("loss_contrast_total", 0.0)
                    recon_raw = logs.get("loss_geo_mae_raw", 0.0)

                    # Loss ratio as proxy for gradient ratio
                    loss_ratio = recon_weighted / (contrast_total + 1e-8)

                    grad_metrics = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss_contrast_total": round(contrast_total, 6),
                        "loss_recon_weighted": round(recon_weighted, 6),
                        "loss_recon_raw": round(recon_raw, 6),
                        "loss_ratio_recon": round(loss_ratio, 6),
                        "mask_ratio_actual": float(cfg.get("data", {}).get("mae", {}).get("mask_ratio", 0.30)),
                    }
                    append_jsonl(grad_log, grad_metrics)

            # Accumulate metrics
            for k, v in logs.items():
                if k == "epoch":
                    continue
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v

            # Print
            if (batch_idx + 1) % print_freq == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                msg = (f"  Epoch {epoch} step {batch_idx + 1}: "
                       f"loss={logs.get('loss_total', 0):.4f} lr={current_lr:.6f}")
                if "loss_geo_mae_raw" in logs:
                    msg += f" geo_mae={logs['loss_geo_mae_raw']:.4f}"
                if "loss_reconstruction_weighted" in logs:
                    msg += f" recon_w={logs['loss_reconstruction_weighted']:.4f}"
                print(msg)

            # Write step-level JSONL
            if (batch_idx + 1) % print_freq == 0:
                step_log = {
                    "epoch": epoch,
                    "step": batch_idx + 1,
                    "lr": optimizer.param_groups[0]["lr"],
                    **logs,
                }
                append_jsonl(train_log, step_log)

            if args.dry_run_batches > 0 and batch_idx >= args.dry_run_batches:
                break

        # Epoch average
        scheduler.step()
        elapsed = time.time() - t0
        avg_metrics = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                       "grad_accum_steps": grad_accum_steps,
                       "epoch_time_s": round(elapsed, 1)}
        for k, v in epoch_metrics.items():
            avg_metrics[k] = v / len(loader)

        append_jsonl(train_log, avg_metrics)
        msg = (f"Epoch {epoch}: loss_total={avg_metrics.get('loss_total', 0):.4f} "
               f"lr={avg_metrics['lr']:.6f} time={elapsed:.0f}s")
        if "loss_geo_mae_raw" in avg_metrics:
            msg += f" geo_mae={avg_metrics['loss_geo_mae_raw']:.4f}"
        print(msg)

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": cfg,
                "best_modelnet": best_mn,
                "best_scan": best_sn,
                "best_balanced": best_balanced,
                "global_step": global_step,
            }, os.path.join(run_dir, f"ckpt_epoch_{epoch}.pth"))

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
                    }, os.path.join(run_dir, "best_modelnet_model.pth"))
                    print(f"  *** New best ModelNet40: {best_mn:.4f}")

                if sn_acc > best_sn:
                    best_sn = sn_acc
                    save_checkpoint({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                    }, os.path.join(run_dir, "best_scan_model.pth"))
                    print(f"  *** New best ScanObjectNN: {best_sn:.4f}")

                if bal_score > best_balanced:
                    best_balanced = bal_score
                    save_checkpoint({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                    }, os.path.join(run_dir, "best_balanced_model.pth"))
                    print(f"  *** New best balanced: {best_balanced:.4f}")

                print(f"{'='*40} End Evaluation {'='*40}\n")

        if args.dry_run_batches > 0 and epoch >= 2:
            break

    # Save final
    save_checkpoint({
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, os.path.join(run_dir, "last_model.pth"))

    print(f"\nTraining complete. Run dir: {run_dir}")
    print(f"Best ModelNet40: {best_mn:.4f}")
    print(f"Best ScanObjectNN: {best_sn:.4f}")
    print(f"Best Balanced: {best_balanced:.4f}")


if __name__ == "__main__":
    main()
