#include "knn.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
constexpr float kInt8Scale = 32.0f;
constexpr float kInt16Scale = 1024.0f;

float absf(float x) { return x < 0 ? -x : x; }

float quantize_to_step(float v, float step) {
  return std::round(v * step) / step;
}

}  // namespace

KNNAccelerator::KNNAccelerator(KNNConfig config) : config_(config) {
  if (config_.k_value == 0) {
    throw std::invalid_argument("k_value must be > 0");
  }
  if (config_.pe_count == 0) {
    throw std::invalid_argument("pe_count must be > 0");
  }
}

void KNNAccelerator::load_training_sample(const Sample& sample) {
  if (config_.num_features != 0 && sample.features.size() != config_.num_features) {
    throw std::invalid_argument("sample feature size does not match num_features");
  }
  if (sample.label < 0 || static_cast<size_t>(sample.label) >= config_.num_classes) {
    throw std::invalid_argument("sample label out of class range");
  }
  training_data_.push_back(sample);
}

float KNNAccelerator::quantize(float value) const {
  switch (config_.precision_mode) {
    case PrecisionMode::Int8:
      return quantize_to_step(value, kInt8Scale);
    case PrecisionMode::Int16:
      return quantize_to_step(value, kInt16Scale);
    case PrecisionMode::Fp16:
    default:
      return quantize_to_step(value, 2048.0f);
  }
}

float KNNAccelerator::distance(const std::vector<float>& a, const std::vector<float>& b,
                               float worst_topk,
                               size_t* processed_features) const {
  const size_t total_features = std::min(a.size(), b.size());
  size_t used_features = total_features;
  if (config_.approx_mode) {
    const float ratio = std::max(0.1f, std::min(1.0f, config_.approx_feature_ratio));
    used_features = std::max<size_t>(1, static_cast<size_t>(std::floor(total_features * ratio)));
  }

  float acc = 0.0f;
  size_t i = 0;
  for (; i < used_features; ++i) {
    const float diff = quantize(a[i]) - quantize(b[i]);
    if (config_.distance_mode == DistanceMode::L1) {
      acc += absf(diff);
    } else {
      acc += diff * diff;
    }

    if (config_.early_exit && worst_topk < std::numeric_limits<float>::infinity() &&
        acc > worst_topk) {
      ++i;
      break;
    }
  }

  if (processed_features) {
    *processed_features = i;
  }

  return acc;
}

std::vector<std::pair<float, int>> KNNAccelerator::hierarchical_topk(
    const std::vector<std::pair<float, int>>& distances) const {
  const size_t k = std::min(config_.k_value, distances.size());
  if (k == 0) return {};

  const size_t chunks = std::min(config_.pe_count, std::max<size_t>(1, distances.size()));
  std::vector<std::vector<std::pair<float, int>>> local(chunks);

  for (size_t i = 0; i < distances.size(); ++i) {
    local[i % chunks].push_back(distances[i]);
  }

  for (auto& v : local) {
    std::partial_sort(v.begin(), v.begin() + std::min(k, v.size()), v.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    if (v.size() > k) v.resize(k);
  }

  std::vector<std::pair<float, int>> merged;
  for (const auto& v : local) {
    merged.insert(merged.end(), v.begin(), v.end());
  }

  std::partial_sort(merged.begin(), merged.begin() + std::min(k, merged.size()),
                    merged.end(),
                    [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  if (merged.size() > k) merged.resize(k);
  return merged;
}

InferenceResult KNNAccelerator::classify(const std::vector<float>& test_features) const {
  if (training_data_.empty()) {
    throw std::runtime_error("no training data loaded");
  }
  if (config_.num_features != 0 && test_features.size() != config_.num_features) {
    throw std::invalid_argument("test feature size does not match num_features");
  }

  std::vector<std::pair<float, int>> distances;
  distances.reserve(training_data_.size());

  float worst = std::numeric_limits<float>::infinity();
  std::vector<float> running_topk;
  running_topk.reserve(config_.k_value);

  for (const auto& sample : training_data_) {
    float d = distance(test_features, sample.features, worst, nullptr);
    distances.push_back({d, sample.label});

    if (running_topk.size() < config_.k_value) {
      running_topk.push_back(d);
      if (running_topk.size() == config_.k_value) {
        worst = *std::max_element(running_topk.begin(), running_topk.end());
      }
    } else if (d < worst) {
      auto it = std::max_element(running_topk.begin(), running_topk.end());
      *it = d;
      worst = *std::max_element(running_topk.begin(), running_topk.end());
    }
  }

  auto topk = hierarchical_topk(distances);
  return vote_topk(topk, config_.num_classes);
}

std::vector<std::pair<float, int>> compute_distances_golden(
    const KNNConfig& config, const std::vector<Sample>& train,
    const std::vector<float>& test) {
  std::vector<std::pair<float, int>> out;
  out.reserve(train.size());

  for (const auto& s : train) {
    float acc = 0.0f;
    for (size_t i = 0; i < std::min(test.size(), s.features.size()); ++i) {
      const float diff = test[i] - s.features[i];
      if (config.distance_mode == DistanceMode::L1) {
        acc += diff < 0 ? -diff : diff;
      } else {
        acc += diff * diff;
      }
    }
    out.push_back({acc, s.label});
  }

  std::sort(out.begin(), out.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  if (out.size() > config.k_value) out.resize(config.k_value);
  return out;
}
