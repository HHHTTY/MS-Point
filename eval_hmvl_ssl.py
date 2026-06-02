import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import yaml
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from datasets.data1 import ModelNet40SVM, ScanObjectNNSVM
from hmvl_experiments.model import HierarchicalCrossPoint


def extract_features(model, dataset_name, partition, args, device, feature_name="feat", batch_size=128, num_workers=8):
    dataset_cls = ModelNet40SVM if dataset_name == "ModelNet40" else ScanObjectNNSVM
    dataset = dataset_cls(partition=partition, num_points=args.num_points)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for data, label in loader:
            data = data.to(device, non_blocking=True)
            encoded = model.encode_point_tensor(data, "global")
            feats.append(encoded[feature_name].detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy().reshape(-1))
    import numpy as np
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--c_values", nargs="+", type=float, default=None)
    args_cli = parser.parse_args()

    ckpt = torch.load(args_cli.checkpoint, map_location="cpu")
    cfg = ckpt.get("config")
    if cfg is None:
        cfg_path = Path(args_cli.checkpoint).with_name("config.yaml")
        cfg = yaml.safe_load(cfg_path.read_text())
    model_args = SimpleNamespace(
        num_points=int(cfg["model"].get("num_points", 1024)),
        emb_dims=int(cfg["model"].get("emb_dims", 1024)),
        k=int(cfg["model"].get("k", 15)),
        dropout=float(cfg["model"].get("dropout", 0.5)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalCrossPoint(model_args, cfg["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    eval_cfg = cfg.get("eval", {})
    datasets = args_cli.datasets or eval_cfg.get("datasets", ["ModelNet40", "ScanObjectNN"])
    features = args_cli.features or eval_cfg.get("features", ["feat", "proj"])
    c_values = args_cli.c_values or eval_cfg.get("c_values", [0.005, 0.01, 0.02, 0.03, 0.05, 0.1])
    rows = []
    for dataset_name in datasets:
        for feature_name in features:
            x_train, y_train = extract_features(model, dataset_name, "train", model_args, device, feature_name=feature_name, batch_size=int(eval_cfg.get("batch_size", 128)), num_workers=int(eval_cfg.get("num_workers", 8)))
            x_test, y_test = extract_features(model, dataset_name, "test", model_args, device, feature_name=feature_name, batch_size=int(eval_cfg.get("batch_size", 128)), num_workers=int(eval_cfg.get("num_workers", 8)))
            for c in c_values:
                clf = SVC(C=float(c), kernel="linear", cache_size=4096)
                clf.fit(x_train, y_train)
                rows.append({"dataset": dataset_name, "feature": feature_name, "C": float(c), "accuracy": float(clf.score(x_test, y_test)), "checkpoint": args_cli.checkpoint})
    df = pd.DataFrame(rows)
    out = Path(args_cli.out) if args_cli.out else Path(args_cli.checkpoint).with_suffix(".eval.csv")
    df.to_csv(out, index=False)
    print(df)
    print("Saved", out)


if __name__ == "__main__":
    main()
