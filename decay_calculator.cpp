#include "decay_calculator.h"

#include <cmath>

std::unordered_map<float, float> DecayCalculator::minimum_duration_map;
std::unordered_map<float, const float*>
    DecayCalculator::precalculated_factors_map;

DecayCalculator::DecayCalculator(const float decay_rate_) :
  decay_rate(decay_rate_),
  minimum_duration(calculate_minimum_duration(decay_rate_)),
  precalculated_factors(
      calculate_precalculated_factors(minimum_duration, decay_rate_)),
  previous_timestamp(0)
{
}

// Returns the decay factor for the specified duration and decay rate.
static float decay_factor_for_duration(
    const float duration,
    const float decay_rate
) {
  return expf(duration * decay_rate);
}

bool DecayCalculator::calculate_factor(const float timestamp, float* factor) {
  const float duration = timestamp - previous_timestamp;
  if (duration < minimum_duration) {
    return false;
  }
  const int milliseconds = (duration - minimum_duration) * 1000;
  if (milliseconds < PRECALCULATED_FACTOR_COUNT) {
    *factor = precalculated_factors[milliseconds];
  } else {
    *factor = decay_factor_for_duration(duration, decay_rate);
  }
  previous_timestamp = timestamp;
  return true;
}

void DecayCalculator::reset() {
  previous_timestamp = 0;
}

float DecayCalculator::calculate_minimum_duration(const float decay_rate) {
  // Check for a cached result.
  const auto it = minimum_duration_map.find(decay_rate);
  if (it != minimum_duration_map.end()) {
    return it->second;
  }

  // Calculate the duration at which decay becomes meaningful.
  constexpr float decay_threshold = 127.0f / 128.0f;
  float minimum_duration = 0;
  for (unsigned int i = 1;; i++) {
    const float duration = i * 1e-3;
    if (decay_factor_for_duration(duration, decay_rate) < decay_threshold) {
      minimum_duration = duration;
      break;
    }
  }
  minimum_duration_map[decay_rate] = minimum_duration;
  return minimum_duration;
}

const float* DecayCalculator::calculate_precalculated_factors(
    const float minimum_duration,
    const float decay_rate
) {
  // Check for a cached result.
  const auto it = precalculated_factors_map.find(decay_rate);
  if (it != precalculated_factors_map.end()) {
    return it->second;
  }

  // Pre-calculate some decay factors.
  float* precalculated_factors = new float[PRECALCULATED_FACTOR_COUNT];
  for (unsigned int i = 0; i < PRECALCULATED_FACTOR_COUNT; i++) {
    precalculated_factors[i] = decay_factor_for_duration(
        minimum_duration + i * 1e-3, decay_rate);
  }
  precalculated_factors_map[decay_rate] = precalculated_factors;
  return precalculated_factors;
}
