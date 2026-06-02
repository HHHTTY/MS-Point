
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from datasets.data1 import ModelNet40SVM, ScanObjectNNSVM
from mvms_experiments.model import MultiViewCrossPoint


def extract(model, dataset_name, partition, args, device, feature, batch_size=128):
    cls = ModelNet40SVM if dataset_name == "ModelNet40" else ScanObjectNNSVM
    loader = DataLoader(cls(partition=partition, num_points=args.num_points), batch_size=batch_size, shuffle=False, num_workers=8)
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            encoded = model.encode_point_tensor(data, "global")
            feats.append(encoded[feature].detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy().reshape(-1))
    return np.concatenate(feats), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--c_values", default="0.005,0.01,0.02,0.03,0.04,0.05,0.1,0.2")
    args_cli = parser.parse_args()
    ckpt = torch.load(args_cli.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    args = SimpleNamespace(
        num_points=int(cfg["model"].get("num_points", 1024)),
        emb_dims=int(cfg["model"].get("emb_dims", 1024)),
        k=int(cfg["model"].get("k", 15)),
        dropout=float(cfg["model"].get("dropout", 0.5)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiViewCrossPoint(args, cfg["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    c_values = [float(x) for x in args_cli.c_values.split(",")]
    rows = []
    for dataset_name in ["ModelNet40", "ScanObjectNN"]:
        for feature in ["feat", "proj"]:
            xtr, ytr = extract(model, dataset_name, "train", args, device, feature)
            xte, yte = extract(model, dataset_name, "test", args, device, feature)
            for c in c_values:
                clf = SVC(C=c, kernel="linear", cache_size=4096)
                clf.fit(xtr, ytr)
                acc = clf.score(xte, yte)
                rows.append({"dataset": dataset_name, "feature": feature, "C": c, "accuracy": float(acc), "checkpoint": args_cli.checkpoint})
                print(rows[-1])
    df = pd.DataFrame(rows)
    out = Path(args_cli.out) if args_cli.out else Path(args_cli.checkpoint).with_suffix(".eval.csv")
    df.to_csv(out, index=False)
    print("Saved", out)


if __name__ == "__main__":
    main()
