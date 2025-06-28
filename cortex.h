#ifndef _cortex_h
#define _cortex_h

#include "neuron.h"

#include <vector>

// The cortex interface.
class Cortex {
  public:
    // Creates a neuron and adds it to the cortex.
    void add_neuron(
        uint16_t output_channel,
        uint16_t num_channels,
        const int8_t* weights,
        const Parameters& parameters);

    // Sends a spike to the specified input channel.
    // Returns a list of the output channels that fire as a result.
    void spike(
        float timestamp,
        uint16_t input_channel,
        std::vector<uint16_t>* outputs);

    // Resets the activation level of all the neurons.
    void reset();

    // Reserves storage for the specified number of neurons.
    void reserve(unsigned int num_neurons) {
      neurons.reserve(num_neurons);
    }

    // Returns the number of neurons.
    unsigned int neuron_count() const { return neurons.size(); }

  private:
    // The neurons in the cortex.
    std::vector<Neuron> neurons;
};

#endif // _cortex_h
