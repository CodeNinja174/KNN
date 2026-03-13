# Publishable Methodology Package

This repository now contains an executable reference implementation and reproducible experiment scripts aligned to the PRD.

## What is implemented
- Parameterized KNN accelerator software model (`KNNConfig`) with:
  - runtime control fields for K, dataset sizes, feature count, class count
  - distance mode (`L1`, `L2`)
  - precision mode (`Int8`, `Int16`, `Fp16` quantized simulation)
  - approximate mode (feature ratio pruning)
  - early-exit distance compute
  - PE-count-aware hierarchical top-k merge model
- Testbench that validates software model prediction against golden KNN.
- Dataset generation and ablation scripts that emit machine-readable CSV.

## Reproducibility steps
1. `python3 scripts/generate_dataset.py`
2. `make run`
3. `python3 scripts/run_experiments.py`
4. Attach `results/ablation_results.csv` to manuscript artifacts.

## Manuscript structure recommendation
1. Introduction and motivation.
2. Architecture and module details.
3. Experimental setup (datasets, tool versions, platform).
4. Ablation study.
5. Baseline comparisons (CPU/GPU/FPGA prior work).
6. Threats to validity and limitations.
