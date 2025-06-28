#include "hippocampus.h"

Hippocampus::Hippocampus(
    const uint16_t num_channels_,
    const Parameters& parameters
) :
  num_channels(num_channels_),
  cumulative_inputs(new DecayingValue*[num_channels])
{
  for (uint16_t i = 0; i < num_channels; i++) {
    cumulative_inputs[i] = new DecayingValue(
        parameters.DECAY_HALF_LIFE, parameters.SPIKE_FRACTION);
  }
  for (uint16_t i = 0; i < num_channels; i++) {
    channels.emplace_back(
        /* channel= */ i,
        parameters.NEGATIVE_WEIGHT_HALF_LIFE,
        parameters.NEGATIVE_SPIKE_FRACTION);
  }
}

Hippocampus::~Hippocampus() {
  for (uint16_t i = 0; i < num_channels; i++) {
    delete cumulative_inputs[i];
  }
  delete[] cumulative_inputs;
}

void Hippocampus::receive_input(
    const float timestamp,
    const uint16_t input_channel,
    const Parameters& parameters,
    Cortex* cortex,
    std::vector<uint16_t>* outputs
) {
  // Apply the weighted spike to all the under-construction neurons.
  const int8_t weighted_input =
      cumulative_inputs[input_channel]->get_weight(timestamp);
  if (weighted_input > 0) {
    for (HCChannel& channel : channels) {
      if (!channel.activate(timestamp, weighted_input)) {
        continue;
      }
      outputs->push_back(channel.get_id());
      if (!channel.should_create_neuron()) {
        continue;
      }
      // Add the under-construction neuron to the cortex.
      int8_t weights[num_channels];
      for (uint16_t i = 0; i < num_channels; i++) {
        weights[i] = cumulative_inputs[i]->get_weight(timestamp)
            + channel.calculate_negative_weight(timestamp);
      }
      cortex->add_neuron(channel.get_id(), num_channels, weights, parameters);
      channel.reset();
    }
  }

  // Spike the cumulative inputs to update the weight of the input channel.
  cumulative_inputs[input_channel]->spike(timestamp);

  // Indicate an input on the hippocampus channel.
  channels[input_channel].receive_input(timestamp);
}

void Hippocampus::receive_output(
    const float timestamp,
    const uint16_t output_channel
) {
  // Indicate an output on the hippocampus channel.
  channels[output_channel].receive_output(timestamp);
}

void Hippocampus::reset() {
  for (uint16_t i = 0; i < num_channels; i++) {
    cumulative_inputs[i]->reset();
  }
  for (HCChannel& channel : channels) {
    channel.reset();
  }
}
