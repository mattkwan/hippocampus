#include "neuron.h"

#include <cstring>

Neuron::Neuron(
    const uint16_t output_channel_,
    const uint16_t num_channels_,
    const int8_t* weights_,
    const Parameters& parameters
) :
  output_channel(output_channel_),
  activation_level(0),
  refractory_period_end_time(0),
  refractory_duration(parameters.MIN_SPIKE_INTERVAL),
  num_channels(num_channels_),
  weights(new int8_t[num_channels])
{
  memcpy(weights, weights_, num_channels * sizeof(int8_t));
}

Neuron::Neuron(const Neuron& neuron) :
  output_channel(neuron.output_channel),
  activation_level(0),
  refractory_period_end_time(0),
  refractory_duration(neuron.refractory_duration),
  num_channels(neuron.num_channels),
  weights(new int8_t[num_channels])
{
  memcpy(weights, neuron.weights, num_channels * sizeof(int8_t));
}

Neuron::~Neuron() {
  delete[] weights;
}

bool Neuron::spike(const float timestamp, const uint16_t input_channel) {
  if (timestamp < refractory_period_end_time) {
    return false;
  }
  activation_level += weights[input_channel];
  if (activation_level >= 128) {
    activation_level = 0;
    refractory_period_end_time = timestamp + refractory_duration;
    return true;
  }
  if (activation_level < 0) {
    activation_level = 0;
  }
  return false;
}

void Neuron::reset() {
  activation_level = 0;
  refractory_period_end_time = 0;
}
