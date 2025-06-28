#include "cortex.h"

void Cortex::spike(
    const float timestamp,
    const uint16_t input_channel,
    std::vector<uint16_t>* outputs
) {
  for (Neuron& neuron : neurons) {
    if (neuron.spike(timestamp, input_channel)) {
      outputs->push_back(neuron.get_output_channel());
    }
  }
}

void Cortex::add_neuron(
    const uint16_t output_channel,
    const uint16_t num_channels,
    const int8_t* weights,
    const Parameters& parameters
) {
  neurons.emplace_back(output_channel, num_channels, weights, parameters);
}

void Cortex::reset() {
  for (Neuron& neuron : neurons) {
    neuron.reset();
  }
}
