#ifndef _brain_h
#define _brain_h

#include "cortex.h"
#include "hippocampus.h"
#include "parameters.h"

#include <vector>

// A processing unit comprising a cerebral cortex and a hippocampus.
class Brain {
  public:
    // Constructor.
    Brain(const uint16_t num_channels, const Parameters& parameters);

    // Reserves storage for the specified number of neurons.
    void reserve(unsigned int num_neurons);

    // Sends a spike to the specified input channel.
    // If use_hippocampus is true, learning is enabled. Otherwise only the
    // cortex is engaged.
    // Returns a list of the output channels that fire as a result.
    void spike(
        float timestamp,
        uint16_t input_channel,
        bool use_hippocamus,
        const Parameters& parameters,
        std::vector<uint16_t>* outputs);

    // Resets the cortex and hippocampus.
    void reset();

    // Returns the number of neurons in the cortex.
    unsigned int neuron_count() const { return cortex.neuron_count(); }

  private:
    // The cerebral cortex.
    Cortex cortex;

    // The hippocampus.
    Hippocampus hippocampus;
};

#endif // _brain_h
