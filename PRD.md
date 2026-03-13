# Parameterized Streaming KNN Accelerator for SoC FPGA
## Product Requirements Document (PRD)

Version: 1.0  
Target Platform: RFSoC / SoC FPGA  
Toolchain: Vitis HLS + Vivado  
Language: C++

---

# 1. Overview

## 1.1 Objective

Design a **parameterized hardware accelerator for the K-Nearest Neighbor (KNN) algorithm** using Vitis HLS targeting SoC FPGA platforms.

The accelerator must:

- Support **runtime configurable datasets**
- Operate with **AXI-Stream input/output**
- Use **AXI-Lite control registers**
- Support **streaming training and testing**
- Implement **parallel distance computation**
- Provide **low-latency Top-K neighbor selection**
- Be **scalable across datasets and feature sizes**

The architecture should be suitable for research publication and reusable for multiple ML workloads.

---

# 2. Research Contributions

The proposed accelerator introduces several architectural contributions:

### 1. Dataset-Agnostic KNN Hardware

The accelerator supports **any dataset without recompilation** using runtime parameters.

Runtime configurable parameters include:

- number of features
- number of training samples
- number of test samples
- number of classes
- K value

---

### 2. Fully Streaming KNN Architecture

Most FPGA KNN implementations store the entire training dataset in on-chip memory.

This design instead uses:

- streaming input
- external memory buffering
- partial on-chip storage

Benefits:

- supports large datasets
- scalable to real workloads
- efficient for SoC FPGA data pipelines

---

### 3. Hierarchical Top-K Selection Engine

Instead of sorting all distances, the architecture uses:

- local Top-K selection
- hierarchical merging

This significantly reduces hardware cost and latency.

---

### 4. Parallel Distance Engine Array

Multiple processing elements (PEs) compute distances simultaneously.


Throughput ∝ Number of PEs


---

### 5. Early-Exit Distance Computation

Distance computation can terminate early if the partial distance exceeds the worst Top-K value.

Benefits:

- significant reduction in computation
- improves latency for large feature sets

---

### 6. Multi-Precision Support

Distance engines support multiple data types:


int8
int16
float16


Precision can be selected via control registers.

---

# 3. System Architecture

## 3.1 SoC System Integration

The accelerator is integrated into the programmable logic (PL) of a SoC FPGA.


+----------------------+
| ARM Processing System|
+----------+-----------+
|
AXI Lite
|
+----------v-----------+
| KNN Accelerator |
| (PL) |
+----------+-----------+
|
AXI Stream
|
DMA / DDR


---

# 4. Accelerator Operation

## 4.1 Modes

The accelerator supports two modes.

### Mode 0 — Training Load

Training samples are streamed and stored internally.

Input format:


[f1 f2 f3 ... fn label]


---

### Mode 1 — Test and Classification

Test vectors are streamed and classified.

For each test vector:

1. compute distances
2. select k nearest neighbors
3. perform voting
4. output predicted class

Input format:


[f1 f2 f3 ... fn]


Output format:


[class_id]


---

# 5. AXI-Lite Register Map

| Address | Register | Description |
|--------|--------|-------------|
| 0x00 | control | start / reset |
| 0x04 | mode | 0=train, 1=test |
| 0x08 | k_value | number of neighbors |
| 0x0C | num_train | number of training samples |
| 0x10 | num_test | number of test samples |
| 0x14 | num_features | feature dimension |
| 0x18 | num_classes | number of classes |
| 0x1C | distance_mode | 0=L2, 1=L1 |
| 0x20 | precision_mode | int8/int16/fp16 |
| 0x24 | approx_mode | enable approximate KNN |

---

# 6. AXI Stream Interface

## 6.1 Input Stream

Training mode packet:


[f1 f2 f3 ... fn label]


Test mode packet:


[f1 f2 f3 ... fn]


---

## 6.2 Output Stream


[class_id]


Optional extended output:


[class_id confidence]


---

# 7. Hardware Architecture

## Top Level Architecture

             +-------------------+

AXI-Lite ------->| Control Registers |
+---------+---------+
|
v
+-------------------+
| Controller FSM |
+---------+---------+
|
v
+-------------------+
AXI Stream ----->| Feature Buffer |
+---------+---------+
|
v
+--------------------------------+
| Parallel Distance Engine Array |
| |
| PE0 PE1 PE2 ... PEn |
+-----------+--------------------+
|
v
+-------------+
| Top-K Unit |
+-------------+
|
v
+-------------+
| Voting Unit |
+-------------+
|
v
AXI Stream Out


---

# 8. Hardware Modules

---

# 8.1 Feature Buffer

Stores training samples streamed from the host.

Implementation:

- BRAM
- partitioned HLS arrays

Structure:


train_data[num_train][num_features]
train_label[num_train]


---

# 8.2 Distance Engine

Computes the distance between test vector and training vector.

Supported functions:

### Euclidean Distance


distance = Σ (test_i − train_i)^2


### Manhattan Distance


distance = Σ |test_i − train_i|


Pipeline stages:


SUB
ABS / MUL
ACCUMULATE


---

# 8.3 Parallel Distance Engine Array

Multiple distance engines operate simultaneously.

Parameter:


PE_COUNT


Training samples are distributed across PEs.

---

# 8.4 Hierarchical Top-K Selection

Traditional sorting has O(N log N) complexity.

The proposed method uses:


Distance stream
↓
Local Top-K
↓
Global Top-K merge


Benefits:

- reduced latency
- lower hardware cost

Registers maintained:


dist[k]
label[k]


---

# 8.5 Voting Unit

Determines the predicted class.

Steps:

1. count occurrences of labels among K neighbors
2. select class with maximum votes


prediction = argmax(votes)


---

# 9. Novel Optimization Techniques

## 9.1 Streaming Training Window

Instead of storing the entire dataset:


DDR → stream → compute → discard


This enables classification on **datasets larger than on-chip memory**.

---

## 9.2 Early Exit Distance Computation

During distance calculation:


partial_distance += diff^2

if partial_distance > worst_topk
terminate computation


Benefits:

- reduces operations
- large speedup for high-dimensional data

---

## 9.3 Approximate KNN Mode

Approximation reduces compute by:

- feature pruning
- partial distance thresholding

Register:


approx_mode


---

# 10. HLS Implementation Guidelines

## 10.1 Pipeline


#pragma HLS PIPELINE


Used in distance loops.

---

## 10.2 Loop Unrolling


#pragma HLS UNROLL


Used for feature-level parallelism.

---

## 10.3 Memory Partitioning


#pragma HLS ARRAY_PARTITION


Allows parallel memory access.

---

## 10.4 Dataflow


#pragma HLS DATAFLOW


Separates pipeline stages:

- input
- compute
- top-k
- voting

---

# 11. Performance Targets

| Metric | Target |
|------|------|
| Clock | 200 MHz |
| Latency | <1 ms per sample |
| Throughput | >1M distances/s |
| BRAM usage | <50% |
| LUT usage | <50% |

---

# 12. Datasets for Evaluation

Recommended evaluation datasets:

- MNIST
- Iris
- CIFAR-10

Metrics measured:

- classification accuracy
- latency
- throughput
- resource utilization
- energy efficiency

---

# 13. Repository Structure


knn_hls_accelerator/

README.md

docs/
PRD.md
architecture.md
diagrams/

src/
knn_top.cpp
distance_engine.cpp
topk.cpp
voting.cpp
controller.cpp

include/
knn.hpp
parameters.hpp

tb/
tb_knn.cpp
dataset_loader.cpp
golden_knn.cpp

scripts/
generate_dataset.py
run_hls.tcl

datasets/
mnist/
iris/


---

# 14. Testbench Requirements

The testbench must:

1. load training dataset
2. stream training data
3. stream test data
4. compute golden output using CPU KNN
5. compare predictions

---

# 15. Experimental Evaluation

The following results should be reported in the paper.

| Metric | Description |
|------|-------------|
| latency | time per classification |
| throughput | samples per second |
| FPGA resources | LUT, FF, BRAM, DSP |
| energy efficiency | joules per inference |
| classification accuracy | dataset accuracy |

Comparisons should be made against:

- CPU KNN
- GPU KNN
- existing FPGA KNN accelerators

---

# 16. Development Milestones

| Week | Task |
|------|------|
| 1 | architecture design |
| 2 | distance engine |
| 3 | PE array |
| 4 | Top-K unit |
| 5 | voting unit |
| 6 | AXI integration |
| 7 | HLS optimization |
| 8 | experiments and paper writing |

---

# 17. Expected Outcome

The final system will provide:

- scalable hardware KNN
- runtime configurable datasets
- high throughput classification
- efficient FPGA resource usage

The architecture will serve as the basis for a research publication in FPGA acceleration for machine learning.

---
