#include "knn.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

static bool nearly_equal(float a, float b, float eps = 1e-4f) {
  return std::fabs(a - b) <= eps;
}

int main() {
  try {
    KNNConfig cfg;
    cfg.k_value = 3;
    cfg.num_features = 4;
    cfg.num_classes = 3;
    cfg.pe_count = 4;
    cfg.distance_mode = DistanceMode::L2;
    cfg.precision_mode = PrecisionMode::Fp16;

    KNNAccelerator accel(cfg);

    std::vector<Sample> train = {
        {{5.1f, 3.5f, 1.4f, 0.2f}, 0}, {{4.9f, 3.0f, 1.4f, 0.2f}, 0},
        {{5.8f, 2.7f, 5.1f, 1.9f}, 2}, {{6.0f, 2.2f, 5.0f, 1.5f}, 2},
        {{6.4f, 3.2f, 4.5f, 1.5f}, 1}, {{6.9f, 3.1f, 4.9f, 1.5f}, 1}};

    for (const auto& s : train) accel.load_training_sample(s);

    std::vector<float> q = {6.1f, 2.9f, 4.7f, 1.4f};
    InferenceResult hw = accel.classify(q);

    auto golden_topk = compute_distances_golden(cfg, train, q);
    InferenceResult golden = vote_topk(golden_topk, cfg.num_classes);

    if (hw.predicted_class != golden.predicted_class) {
      std::cerr << "Prediction mismatch. expected=" << golden.predicted_class
                << " got=" << hw.predicted_class << "\n";
      return 1;
    }

    if (!nearly_equal(hw.confidence, golden.confidence, 0.34f)) {
      std::cerr << "Confidence drift too large. expected=" << golden.confidence
                << " got=" << hw.confidence << "\n";
      return 1;
    }

    // Exercise approximate + early-exit path for stability.
    cfg.approx_mode = true;
    cfg.early_exit = true;
    cfg.approx_feature_ratio = 0.75f;
    KNNAccelerator accel_opt(cfg);
    for (const auto& s : train) accel_opt.load_training_sample(s);

    auto opt = accel_opt.classify(q);
    if (opt.predicted_class < 0 || opt.predicted_class >= 3) {
      std::cerr << "Invalid class ID in optimized mode\n";
      return 1;
    }

    std::cout << "PASS: predicted class=" << hw.predicted_class
              << ", confidence=" << hw.confidence << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Testbench exception: " << ex.what() << "\n";
    return 2;
  }
}
