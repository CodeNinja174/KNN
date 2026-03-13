# Publishable Research Execution Plan

## 1) Research framing and publication goals

### 1.1 Core research questions
1. Can a runtime-parameterized, fully streaming KNN accelerator on SoC FPGA match or exceed prior FPGA KNN designs in throughput-per-resource while maintaining accuracy?
2. What is the impact of hierarchical Top-K and early-exit distance computation on latency, energy, and scalability?
3. How does precision selection (int8/int16/fp16) affect the Pareto frontier of accuracy vs. performance vs. resource usage?

### 1.2 Target publication claims
- **Claim A:** Dataset-agnostic KNN hardware without recompilation using AXI-Lite runtime controls.
- **Claim B:** Streaming architecture scales beyond on-chip memory limits with practical DDR-driven workloads.
- **Claim C:** Hierarchical Top-K + PE parallelism + early exit produce measurable latency and throughput improvements at controlled resource cost.
- **Claim D:** Multi-precision mode enables tunable performance/accuracy trade-offs.

### 1.3 Paper-ready hypotheses
- H1: Increasing `PE_COUNT` improves throughput near-linearly up to memory-interface limits.
- H2: Early-exit reduces average MAC-equivalent operations significantly for high-dimensional inputs.
- H3: Hierarchical Top-K lowers critical-path and area overhead compared to full sort.
- H4: int8 and int16 modes improve throughput/energy with bounded accuracy degradation relative to fp16.

---

## 2) Implementation workplan (aligned to PRD architecture)

### Phase A — Baseline correctness (Weeks 1–2)
**Objective:** Build an end-to-end functionally correct baseline.

Tasks:
1. Implement `knn_top` control flow with AXI-Lite register decode and mode dispatch.
2. Implement mode 0 (training load) and mode 1 (test/classify) stream parsing.
3. Implement single-PE distance kernel (L2 and L1).
4. Implement simple Top-K baseline (reference version) and majority voting.
5. Create C-simulation testbench with CPU golden KNN comparison.

Deliverables:
- Passing functional tests on Iris and synthetic datasets.
- Deterministic output equivalence (or bounded tie-breaking policy).

### Phase B — Throughput architecture (Weeks 3–4)
**Objective:** Introduce performance-critical modules.

Tasks:
1. Implement PE array with parameterized `PE_COUNT`.
2. Partition training buffers for parallel memory access.
3. Implement local Top-K per PE and global hierarchical merge.
4. Add `#pragma HLS DATAFLOW` stage boundaries (ingest/compute/topk/vote).

Deliverables:
- C/RTL co-sim functional pass for multi-PE configs.
- First throughput scaling curves vs. `PE_COUNT`.

### Phase C — Optimization and configurability (Weeks 5–6)
**Objective:** Convert architecture into publishable contribution set.

Tasks:
1. Implement early-exit distance computation with worst-Top-K threshold.
2. Add `precision_mode` support (int8/int16/fp16 path selection).
3. Add `approx_mode` controls (feature pruning / threshold shortcuts).
4. Tune pragmas (`PIPELINE`, `UNROLL`, `ARRAY_PARTITION`) for II and timing.

Deliverables:
- Ablation-ready toggles for each optimization.
- Stable build scripts for batch experiments.

### Phase D — System integration and benchmarking (Weeks 7–8)
**Objective:** Produce publication-grade evidence.

Tasks:
1. Integrate DMA/DDR streaming path on RFSoC/SoC FPGA platform.
2. Validate AXI-Stream packet formats and AXI-Lite runtime control.
3. Run dataset sweeps (Iris, MNIST subsets, CIFAR-10 feature vectors).
4. Collect latency, throughput, LUT/FF/BRAM/DSP, power/energy.
5. Compare against CPU and GPU KNN baselines.

Deliverables:
- Reproducible benchmark table and scripts.
- Figures for speedup, efficiency, and accuracy/resource Pareto frontiers.

---

## 3) Experimental design for publishable evidence

### 3.1 Independent variables
- `PE_COUNT` (e.g., 1, 2, 4, 8, 16)
- `k_value` (e.g., 1, 3, 5, 7)
- `num_features` (low/medium/high dimensions)
- `precision_mode` (int8/int16/fp16)
- `distance_mode` (L1/L2)
- `approx_mode` (off/on)
- Early-exit (off/on)

### 3.2 Dependent metrics
- Accuracy (%)
- Latency per test sample (us/ms)
- Throughput (samples/s and distances/s)
- Resource utilization (LUT, FF, BRAM, DSP)
- Power and energy per inference (J/inference)
- Effective operations saved by early-exit (%)

### 3.3 Mandatory ablations
1. Baseline vs +PE array
2. +PE array vs +hierarchical Top-K
3. +hierarchical Top-K vs +early-exit
4. fp16 vs int16 vs int8 (accuracy and energy impact)
5. exact mode vs approximate mode

### 3.4 Baseline comparisons
- CPU KNN (single-thread + multi-thread variants)
- GPU KNN (standard library implementation)
- Prior FPGA KNN papers (normalized by technology node/frequency where needed)

---

## 4) Reproducibility protocol

1. Fix random seeds for data split and test ordering.
2. Freeze toolchain version (Vitis HLS + Vivado version tags).
3. Store all runtime register configurations per experiment.
4. Record commit hash with each results artifact.
5. Export machine-readable CSV/JSON for all measurements.
6. Provide scripts to regenerate tables and plots in one command.

---

## 5) Risk register and mitigation

1. **Memory bandwidth bottleneck**
   - Mitigation: burst-aligned transfers, double buffering, PE scaling limited by bandwidth roofline.
2. **Timing closure failure at 200 MHz**
   - Mitigation: pipeline balancing, reduced unroll factors, floorplanning guidance.
3. **Accuracy drop in int8 / approximate mode**
   - Mitigation: per-dataset calibration, mixed-precision fallback, report bounded degradation.
4. **Top-K merge critical path growth**
   - Mitigation: multi-stage merge tree, register insertion.
5. **Non-reproducible benchmarking**
   - Mitigation: scripted runs, fixed software stack, version-locked datasets.

---

## 6) Concrete weekly execution checklist

### Week 1
- Finalize module interfaces and register map contract.
- Build CPU golden KNN and basic dataset loader.

### Week 2
- Complete baseline streaming train/test flow and single-PE correctness.

### Week 3
- Implement PE array and data partitioning.

### Week 4
- Implement hierarchical Top-K and verify equivalence with baseline Top-K.

### Week 5
- Add early-exit and quantify operation reduction.

### Week 6
- Add precision and approximate modes; perform calibration.

### Week 7
- Integrate full SoC path (DMA/DDR) and run end-to-end hardware tests.

### Week 8
- Execute full experiment matrix, generate plots/tables, draft paper sections.

---

## 7) Definition of done (publishable threshold)

The work is considered publication-ready when all conditions below are met:
1. Functional correctness verified against CPU golden on all target datasets.
2. Full ablation study completed with statistically stable measurements.
3. Hardware metrics collected from post-implementation reports, not only HLS estimates.
4. At least one clear Pareto-optimal region demonstrated (performance/energy/resource).
5. Reproducibility package (scripts + configs + raw data) is complete.
6. Paper draft includes architecture figure, algorithm pseudocode, methodology, comparisons, and limitations.
