#include "brain.h"

Brain::Brain(const uint16_t num_channels, const Parameters& parameters) :
  hippocampus(num_channels, parameters) {
}

void Brain::spike(
    const float timestamp,
    const uint16_t input_channel,
    const bool use_hippocampus,
    const Parameters& parameters,
    std::vector<uint16_t>* outputs
) {
  // Send the spike to the cortex and collect the output spike channels.
  cortex.spike(timestamp, input_channel, outputs);

  if (use_hippocampus) {
    // Activate the under-construction neurons in the hippocampus and collect
    // the outputs. Also add neurons to the cortex if any become permanent.
    hippocampus.receive_input(
        timestamp, input_channel, parameters, &cortex, outputs);

    // Train the hippocampus on the desired output.
    for (const uint16_t channel : *outputs) {
      hippocampus.receive_output(timestamp, channel);
    }
  }
}

void Brain::reserve(unsigned int num_neurons) {
  cortex.reserve(num_neurons);
}

void Brain::reset() {
  hippocampus.reset();
  cortex.reset();
}
