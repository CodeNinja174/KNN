# Parameterized Streaming KNN Accelerator (Research Scaffold)

This repository contains a **publishable research scaffold** for a parameterized, streaming-style KNN accelerator aligned with `PRD.md`.

## Repository layout

- `include/knn.hpp`: public accelerator configuration + API
- `src/`: software reference model and testable baseline implementation
- `hls/`: synthesizable HLS kernel (`knn_hls_top`) with AXIS + AXI-Lite interfaces and optimization pragmas
- `tb/tb_knn.cpp`: executable correctness testbench vs golden KNN
- `scripts/generate_dataset.py`: synthetic dataset generator
- `scripts/run_experiments.py`: reproducible ablation runner (CSV output)
- `docs/publishable_methodology.md`: reproducibility and paper workflow
- `RESEARCH_EXECUTION_PLAN.md`: high-level execution roadmap

## Quick start

```bash
make run
python3 scripts/generate_dataset.py
python3 scripts/run_experiments.py
cat results/ablation_results.csv
```

## Current status

Implemented features:
- Runtime-parameterized KNN configuration
- L1/L2 distance modes
- Precision-path simulation (int8/int16/fp16 quantization)
- Approximate mode (feature pruning ratio)
- Early-exit distance accumulation
- Hierarchical top-k emulation using PE chunking
- Golden-model validation testbench

This repository includes both a software model and a synthesizable HLS kernel path for low-latency accelerator development.

## HLS synthesis flow

```bash
vitis_hls -f scripts/run_hls.tcl
```

HLS kernel optimization techniques used include DATAFLOW, PIPELINE (II=1), ARRAY_PARTITION, and BRAM binding in `hls/knn_hls_top.cpp`.
