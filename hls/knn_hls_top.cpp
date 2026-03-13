#include "knn_hls.hpp"

static data_t train_mem[MAX_TRAIN][MAX_FEATURES];
static label_t train_lbl[MAX_TRAIN];

static inline ctrl_regs load_ctrl(volatile ap_uint<32>* regs) {
#pragma HLS INLINE
  ctrl_regs c;
  c.start = regs[0];
  c.mode = regs[1];
  c.k_value = regs[2];
  c.num_train = regs[3];
  c.num_test = regs[4];
  c.num_features = regs[5];
  c.num_classes = regs[6];
  c.distance_mode = regs[7];
  c.approx_mode = regs[8];
  c.early_exit = regs[9];
  return c;
}

static inline data_t unpack_feature(packet_t raw) {
#pragma HLS INLINE
  ap_int<16> v = raw.range(15, 0);
  data_t d;
  d.range(15, 0) = v;
  return d;
}

static inline packet_t pack_label(label_t l) {
#pragma HLS INLINE
  packet_t p = 0;
  p.range(7, 0) = l;
  return p;
}

static void load_training(hls::stream<axis_pkt>& s_axis, const ctrl_regs& c) {
#pragma HLS INLINE off
  for (int i = 0; i < c.num_train; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_TRAIN
    for (int f = 0; f < c.num_features; f++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_FEATURES
      axis_pkt pkt = s_axis.read();
      train_mem[i][f] = unpack_feature(pkt.data);
    }
    axis_pkt l = s_axis.read();
    train_lbl[i] = l.data.range(7, 0);
  }
}

static acc_t distance_pe(const data_t test_vec[MAX_FEATURES],
                         const data_t train_vec[MAX_FEATURES], const ctrl_regs& c,
                         acc_t worst_topk) {
#pragma HLS INLINE
  acc_t acc = 0;
  int used_features = c.approx_mode ? (int)c.num_features >> 1 : (int)c.num_features;
  if (used_features < 1) used_features = 1;

DIST_LOOP:
  for (int f = 0; f < used_features; f++) {
#pragma HLS PIPELINE II=1
    data_t d = test_vec[f] - train_vec[f];
    if (c.distance_mode) {
      acc += (d < 0) ? (acc_t)(-d) : (acc_t)d;
    } else {
      acc += (acc_t)(d * d);
    }
    if (c.early_exit && acc > worst_topk) break;
  }
  return acc;
}

static void init_topk(acc_t dist[MAX_K], label_t lbl[MAX_K], ap_uint<8> k) {
#pragma HLS INLINE
  for (int i = 0; i < MAX_K; i++) {
#pragma HLS UNROLL
    if (i < k) {
      dist[i] = (acc_t)1e9;
      lbl[i] = 0;
    }
  }
}

static acc_t update_topk(acc_t cand_d, label_t cand_l, acc_t dist[MAX_K],
                         label_t lbl[MAX_K], ap_uint<8> k) {
#pragma HLS INLINE
  int worst_idx = 0;
  acc_t worst_val = dist[0];

  for (int i = 1; i < MAX_K; i++) {
#pragma HLS UNROLL
    if (i < k && dist[i] > worst_val) {
      worst_val = dist[i];
      worst_idx = i;
    }
  }

  if (cand_d < worst_val) {
    dist[worst_idx] = cand_d;
    lbl[worst_idx] = cand_l;
  }

  // recompute worst for early-exit threshold
  worst_val = dist[0];
  for (int i = 1; i < MAX_K; i++) {
#pragma HLS UNROLL
    if (i < k && dist[i] > worst_val) worst_val = dist[i];
  }
  return worst_val;
}

static label_t vote(label_t topk_lbl[MAX_K], const ctrl_regs& c) {
#pragma HLS INLINE
  ap_uint<16> class_votes[MAX_CLASSES];
#pragma HLS ARRAY_PARTITION variable=class_votes cyclic factor=8
  for (int i = 0; i < MAX_CLASSES; i++) {
#pragma HLS UNROLL
    class_votes[i] = 0;
  }

  for (int i = 0; i < MAX_K; i++) {
#pragma HLS UNROLL
    if (i < c.k_value && topk_lbl[i] < c.num_classes) {
      class_votes[topk_lbl[i]]++;
    }
  }

  label_t best_cls = 0;
  ap_uint<16> best_vote = 0;
  for (int cidx = 0; cidx < MAX_CLASSES; cidx++) {
#pragma HLS UNROLL
    if (cidx < c.num_classes && class_votes[cidx] > best_vote) {
      best_vote = class_votes[cidx];
      best_cls = (label_t)cidx;
    }
  }
  return best_cls;
}

static void classify_stream(hls::stream<axis_pkt>& s_axis, hls::stream<axis_pkt>& m_axis,
                            const ctrl_regs& c) {
#pragma HLS INLINE off
  data_t test_vec[MAX_FEATURES];
#pragma HLS ARRAY_PARTITION variable=test_vec cyclic factor=16

TEST_LOOP:
  for (int t = 0; t < c.num_test; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1024
    for (int f = 0; f < c.num_features; f++) {
#pragma HLS PIPELINE II=1
      axis_pkt pkt = s_axis.read();
      test_vec[f] = unpack_feature(pkt.data);
    }

    acc_t topk_dist[MAX_K];
    label_t topk_lbl[MAX_K];
#pragma HLS ARRAY_PARTITION variable=topk_dist complete
#pragma HLS ARRAY_PARTITION variable=topk_lbl complete
    init_topk(topk_dist, topk_lbl, c.k_value);
    acc_t worst = (acc_t)1e9;

  TRAIN_LOOP:
    for (int i = 0; i < c.num_train; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_TRAIN
      acc_t d = distance_pe(test_vec, train_mem[i], c, worst);
      worst = update_topk(d, train_lbl[i], topk_dist, topk_lbl, c.k_value);
    }

    label_t pred = vote(topk_lbl, c);
    axis_pkt out;
    out.data = pack_label(pred);
    out.last = (t == c.num_test - 1) ? 1 : 0;
    m_axis.write(out);
  }
}

void knn_hls_top(hls::stream<axis_pkt>& s_axis, hls::stream<axis_pkt>& m_axis,
                 volatile ap_uint<32>* regs) {
#pragma HLS INTERFACE axis port=s_axis
#pragma HLS INTERFACE axis port=m_axis
#pragma HLS INTERFACE s_axilite port=regs bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable=train_mem cyclic factor=PE_COUNT dim=1
#pragma HLS BIND_STORAGE variable=train_mem type=RAM_T2P impl=bram
#pragma HLS BIND_STORAGE variable=train_lbl type=RAM_1P impl=bram

  ctrl_regs c = load_ctrl(regs);
  if (!c.start) return;

  if (c.mode == 0) {
    load_training(s_axis, c);
  } else {
    classify_stream(s_axis, m_axis, c);
  }
}
