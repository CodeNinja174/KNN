#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

enum class Mode : uint32_t { Train = 0, Test = 1 };
enum class DistanceMode : uint32_t { L2 = 0, L1 = 1 };
enum class PrecisionMode : uint32_t { Int8 = 0, Int16 = 1, Fp16 = 2 };

struct KNNConfig {
  Mode mode = Mode::Train;
  size_t k_value = 3;
  size_t num_train = 0;
  size_t num_test = 0;
  size_t num_features = 0;
  size_t num_classes = 2;
  DistanceMode distance_mode = DistanceMode::L2;
  PrecisionMode precision_mode = PrecisionMode::Fp16;
  bool approx_mode = false;
  bool early_exit = false;
  size_t pe_count = 1;
  float approx_feature_ratio = 1.0f;
};

struct Sample {
  std::vector<float> features;
  int label = -1;
};

struct InferenceResult {
  int predicted_class = -1;
  float confidence = 0.0f;
};

class KNNAccelerator {
 public:
  explicit KNNAccelerator(KNNConfig config);

  void load_training_sample(const Sample& sample);
  InferenceResult classify(const std::vector<float>& test_features) const;

  const std::vector<Sample>& training_data() const { return training_data_; }
  KNNConfig config() const { return config_; }

 private:
  KNNConfig config_;
  std::vector<Sample> training_data_;

  float quantize(float value) const;
  float distance(const std::vector<float>& a, const std::vector<float>& b,
                 float worst_topk, size_t* processed_features) const;
  std::vector<std::pair<float, int>> hierarchical_topk(
      const std::vector<std::pair<float, int>>& distances) const;
};

std::vector<std::pair<float, int>> compute_distances_golden(
    const KNNConfig& config, const std::vector<Sample>& train,
    const std::vector<float>& test);

InferenceResult vote_topk(const std::vector<std::pair<float, int>>& topk,
                          size_t num_classes);
