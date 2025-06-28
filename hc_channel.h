#ifndef _hc_channel_h
#define _hc_channel_h

#include "decaying_value.h"

// A channel in a hippocampus that creates new neurons.
class HCChannel {
  public:
    // Constructor.
    HCChannel(
        uint16_t id,
        float negative_weight_half_life,
        float negative_weight_spike_fraction);

    // Returns the channel ID.
    uint16_t get_id() const { return id; }

    // Processes an input spike.
    void receive_input(float timestamp);

    // Processes an output spike.
    void receive_output(float timestamp);

    // Activates with a weighted spike. Returns true if the neuron fires.
    bool activate(float timestamp, int8_t weighted_input);

    // Returns true if the under-construction neuron should be added to the
    // cortex.
    bool should_create_neuron() const { return weight_is_correct; }

    // Resets the activation level, fire count, and decay timers.
    void reset();

    // Returns the negative weight that should be applied to all inputs to an
    // under-construction neuron.
    int8_t calculate_negative_weight(float timestamp);

    // Returns the activation level. Visible for testing.
    const int16_t get_activation_level() { return activation_level; }

  private:
    // The channel's ID, a.k.a. the channel number.
    const uint16_t id;

    // The activation level of the under-construction neuron.
    // If the value goes negative, it's clipped to zero.
    // If the value goes >= 128, it's reset to zero and the neuron fires.
    // Technically this is a decaying value, but given the frequency of spikes
    // the decay will be negligible. And calculating decay is expensive.
    int16_t activation_level;

    // The negative weight controller.
    // At zero, the negative weight is most negative.
    // As the controller goes higher, the weight gets closer to zero.
    DecayingValue negative_weight_controller;

    // Whether the negative weight correctly balances inputs and outputs.
    bool weight_is_correct;

    // Fires the under-construction neuron.
    void fire_neuron();
};

#endif // _hc_channel_h
