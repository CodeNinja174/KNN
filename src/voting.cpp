#include "knn.hpp"

#include <algorithm>
#include <vector>

InferenceResult vote_topk(const std::vector<std::pair<float, int>>& topk,
                          size_t num_classes) {
  if (topk.empty() || num_classes == 0) {
    return {-1, 0.0f};
  }

  std::vector<size_t> votes(num_classes, 0);
  for (const auto& item : topk) {
    if (item.second >= 0 && static_cast<size_t>(item.second) < num_classes) {
      votes[static_cast<size_t>(item.second)]++;
    }
  }

  size_t best_cls = 0;
  size_t best_votes = 0;
  for (size_t c = 0; c < votes.size(); ++c) {
    if (votes[c] > best_votes) {
      best_votes = votes[c];
      best_cls = c;
    }
  }

  const float confidence = static_cast<float>(best_votes) / static_cast<float>(topk.size());
  return {static_cast<int>(best_cls), confidence};
}
