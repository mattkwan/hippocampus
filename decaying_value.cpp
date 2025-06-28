#include "decaying_value.h"

#include <cmath>

DecayingValue::DecayingValue(
    const float half_life,
    const float spike_fraction_
) :
  value(0),
  spike_fraction(spike_fraction_),
  decay_calculator(-M_LN2 / half_life)
{
}

float DecayingValue::get_value(const float timestamp) {
  decay_value(timestamp);
  return value;
}

int8_t DecayingValue::get_weight(const float timestamp) {
  const int weight = roundf(get_value(timestamp) * 128.0f);
  return std::min(weight, 127);
}

void DecayingValue::spike(const float timestamp) {
  decay_value(timestamp);
  value += (1.0f - value) * spike_fraction;
}

void DecayingValue::negative_spike(const float timestamp) {
  decay_value(timestamp);
  value *= 1.0f - spike_fraction;
}

void DecayingValue::reset() {
  value = 0;
  decay_calculator.reset();
}

void DecayingValue::decay_value(const float timestamp) {
  float factor;
  if (decay_calculator.calculate_factor(timestamp, &factor)) {
    value *= factor;
  }
}
