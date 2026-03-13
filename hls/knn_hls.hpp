#pragma once

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// High-performance, low-latency KNN HLS kernel parameters.
static constexpr int MAX_FEATURES = 256;
static constexpr int MAX_TRAIN = 4096;
static constexpr int MAX_CLASSES = 32;
static constexpr int MAX_K = 16;
static constexpr int PE_COUNT = 8;

using data_t = ap_fixed<16, 6>;    // fp16-like fixed-point for throughput/resource balance
using acc_t = ap_fixed<32, 12>;
using label_t = ap_uint<8>;
using packet_t = ap_uint<32>;

struct axis_pkt {
  packet_t data;
  ap_uint<1> last;
};

struct ctrl_regs {
  ap_uint<1> start;
  ap_uint<1> mode;  // 0=train, 1=test
  ap_uint<8> k_value;
  ap_uint<16> num_train;
  ap_uint<16> num_test;
  ap_uint<16> num_features;
  ap_uint<8> num_classes;
  ap_uint<1> distance_mode;  // 0=L2, 1=L1
  ap_uint<1> approx_mode;
  ap_uint<1> early_exit;
};

void knn_hls_top(hls::stream<axis_pkt>& s_axis,
                 hls::stream<axis_pkt>& m_axis,
                 volatile ap_uint<32>* regs);
