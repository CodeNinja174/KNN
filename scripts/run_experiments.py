#!/usr/bin/env python3
"""Run reproducible software experiments for ablation planning."""

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class Cfg:
    k: int
    dist: str
    approx: bool
    early_exit: bool
    ratio: float


def dist(a: List[float], b: List[float], mode: str, ratio: float, approx: bool, early: bool, worst: float) -> Tuple[float, int]:
    n = len(a)
    used = max(1, int(n * ratio)) if approx else n
    acc = 0.0
    count = 0
    for i in range(used):
        d = a[i] - b[i]
        acc += abs(d) if mode == "l1" else d * d
        count += 1
        if early and acc > worst:
            break
    return acc, count


def predict(train, q, cfg: Cfg, classes: int):
    pairs = []
    ops = 0
    worst = float("inf")
    running_topk = []
    for feats, label in train:
        d, c = dist(q, feats, cfg.dist, cfg.ratio, cfg.approx, cfg.early_exit, worst)
        ops += c
        pairs.append((d, label))

        if len(running_topk) < cfg.k:
            running_topk.append(d)
            if len(running_topk) == cfg.k:
                worst = max(running_topk)
        elif d < worst:
            idx = running_topk.index(max(running_topk))
            running_topk[idx] = d
            worst = max(running_topk)

    pairs.sort(key=lambda x: x[0])
    topk = pairs[: cfg.k]
    votes = [0] * classes
    for _, lab in topk:
        votes[lab] += 1
    pred = max(range(classes), key=lambda c: votes[c])
    return pred, ops


def load_dataset(path: Path):
    train, test = [], []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            label = int(row["label"])
            feats = [float(row[k]) for k in row if k.startswith("f")]
            (train if row["split"] == "train" else test).append((feats, label))
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/synthetic/synth.csv")
    ap.add_argument("--out", default="results/ablation_results.csv")
    args = ap.parse_args()

    ds = Path(args.dataset)
    if not ds.exists():
        raise SystemExit(f"Dataset not found: {ds}. Run scripts/generate_dataset.py first.")

    train, test = load_dataset(ds)
    classes = len({label for _, label in train})

    configs = [
        Cfg(3, "l2", False, False, 1.0),
        Cfg(3, "l2", False, True, 1.0),
        Cfg(3, "l2", True, False, 0.75),
        Cfg(3, "l1", False, False, 1.0),
    ]

    rows = []
    for cfg in configs:
        t0 = time.perf_counter()
        correct = 0
        ops = 0
        for q, gt in test:
            pred, o = predict(train, q, cfg, classes)
            correct += int(pred == gt)
            ops += o
        dt = time.perf_counter() - t0
        rows.append({
            "k": cfg.k,
            "dist": cfg.dist,
            "approx": int(cfg.approx),
            "early_exit": int(cfg.early_exit),
            "ratio": cfg.ratio,
            "accuracy": round(correct / max(1, len(test)), 4),
            "latency_s": round(dt, 6),
            "avg_feature_ops": round(ops / max(1, len(test)), 2),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out}")


if __name__ == "__main__":
    random.seed(7)
    main()
