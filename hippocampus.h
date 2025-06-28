#ifndef _hippocampus_h
#define _hippocampus_h

#include "cortex.h"
#include "decaying_value.h"
#include "hc_channel.h"
#include "parameters.h"

#include <vector>

// The hippocampus interface.
class Hippocampus {
  public:
    // Constructor.
    Hippocampus(uint16_t num_channels, const Parameters& parameters);

    // Destructor.
    ~Hippocampus();

    // Processes a spike on an input channel.
    // Adds newly-created neurons to the cortex.
    // Returns a list of the output channels that fire as a result.
    void receive_input(
        float timestamp,
        uint16_t input_channel,
        const Parameters& parameters,
        Cortex* cortex,
        std::vector<uint16_t>* outputs);

    // Processes a spike on an output channel.
    void receive_output(float timestamp, uint16_t output_channel);

    // Resets the cumulative inputs and channels.
    void reset();

  private:
    // The number of inputs.
    const uint16_t num_channels;

    // The cumulative input values.
    DecayingValue **cumulative_inputs;

    // The channels in the hippocampus.
    std::vector<HCChannel> channels;
};

#endif // _hippocampus_h
