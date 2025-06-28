#include "hc_channel.h"

#include <cmath>

// The maximum allowed value of the negative weight.
// This value affects how far a weight can decay before it's ignored.
static constexpr int MAX_NEGATIVE_WEIGHT = -4;

HCChannel::HCChannel(
    const uint16_t id_,
    const float negative_weight_half_life,
    const float negative_weight_spike_fraction
) :
  id(id_),
  activation_level(0),
  negative_weight_controller(
      negative_weight_half_life, negative_weight_spike_fraction),
  weight_is_correct(false)
{
}

void HCChannel::receive_input(const float timestamp) {
  negative_weight_controller.spike(timestamp);
  activation_level = 0;
}

void HCChannel::receive_output(const float timestamp) {
  negative_weight_controller.negative_spike(timestamp);
  activation_level = 0;
}

int8_t HCChannel::calculate_negative_weight(const float timestamp) {
  const int negative_weight = roundf(
      (negative_weight_controller.get_value(timestamp) - 1.0f) * 128);
  return std::min(negative_weight, MAX_NEGATIVE_WEIGHT);
}

void HCChannel::reset() {
  activation_level = 0;
  weight_is_correct = false;
  negative_weight_controller.reset();
}

bool HCChannel::activate(
    const float timestamp,
    const int8_t weighted_input
) {
  activation_level += weighted_input + calculate_negative_weight(timestamp);
  if (activation_level >= 128) {  // Causes under-construction neuron to fire.
    fire_neuron();
    return true;
  }
  if (activation_level < 0) {
    // Clip activation level at zero.
    activation_level = 0;
  }
  return false;
}

void HCChannel::fire_neuron() {
  activation_level = 0;
  weight_is_correct = true;
}
