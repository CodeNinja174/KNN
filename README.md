# Parameterized Streaming KNN Accelerator (Research Scaffold)

This repository contains a **publishable research scaffold** for a parameterized, streaming-style KNN accelerator aligned with `PRD.md`.

## Repository layout

- `include/knn.hpp`: public accelerator configuration + API
- `src/`: software reference model and testable baseline implementation
- `hls/`: synthesizable HLS kernel (`knn_hls_top`) with AXIS + AXI-Lite interfaces and optimization pragmas
- `src/`: reference module implementation (`knn_top`, voting, and stubs for module split)
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

This is a **research prototype** designed to support algorithm and architecture studies before RTL/HLS closure on hardware.
