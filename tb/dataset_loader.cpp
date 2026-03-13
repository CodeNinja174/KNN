#include "knn.hpp"

#include <sstream>
#include <string>
#include <vector>

std::vector<float> parse_csv_features(const std::string& line) {
  std::vector<float> values;
  std::stringstream ss(line);
  std::string token;
  while (std::getline(ss, token, ',')) {
    values.push_back(std::stof(token));
  }
  return values;
}
