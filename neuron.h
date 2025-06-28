#ifndef _neuron_h
#define _neuron_h

#include "parameters.h"

#include <cstdint>

// A spiking neuron.
class Neuron {
  public:
    // Constructor for a neuron.
    // The weights are normalized so that a value of 128 will activate the
    // neuron.
    Neuron(
        uint16_t output_channel,
        uint16_t num_channels,
        const int8_t* weights,
        const Parameters& parameters);

    // Disable copy constructor.
    Neuron(const Neuron& neuron);

    // Destructor.
    ~Neuron();

    // Returns the neuron's output channel.
    uint16_t get_output_channel() const { return output_channel; }

    // Sends a spike to the specified input channel. Returns true if the
    // spike causes the neuron to fire.
    bool spike(float timestamp, uint16_t input_channel);

    // Resets the neuron's activation level and refractory period end time.
    void reset();

  private:
    // The channel that the neuron outputs to.
    const uint16_t output_channel;

    // The current activation level.
    // If the value goes negative, it's clipped to zero.
    // If the value goes >= 128, it's reset to zero and the neuron fires.
    // Technically this is an RC-circuit decaying value, but given the
    // frequency of spikes the decay will be negligible. And calculating decay
    // is expensive.
    int16_t activation_level;

    // The time at which the neuron becomes active again.
    // After firing, a neuron remains inactive for a period.
    float refractory_period_end_time;

    // The duration of the refractory period, in seconds.
    const float refractory_duration;

    // The number of weights.
    const uint16_t num_channels;

    // The weights on the input channels.
    int8_t* const weights;
};

#endif // _neuron_h
