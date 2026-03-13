#!/usr/bin/env python3
"""Generate a synthetic dataset for KNN benchmarking."""

import argparse
import csv
import random


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="datasets/synthetic/synth.csv")
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(["split", "label", *[f"f{i}" for i in range(args.features)]])

      for split, count in (("train", args.train), ("test", args.test)):
        for _ in range(count):
          label = random.randrange(args.classes)
          center = label * 3.0
          features = [round(random.gauss(center, 0.8), 4) for _ in range(args.features)]
          writer.writerow([split, label, *features])


if __name__ == "__main__":
    main()
