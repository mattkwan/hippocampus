#ifndef _decaying_value_h
#define _decaying_value_h

#include "decay_calculator.h"

// Maintains a value that can be increased and decreased with spikes, and
// decays exponentially with time.
class DecayingValue {
  public:
    // Constructor.
    // The half life is the time, in seconds, for the value to fall by half.
    // Each spike increases the value by (1 - value) * spike_fraction, moving
    // the value fractionally closer to one.
    DecayingValue(float half_life, float spike_fraction);

    // Returns the value at the specified time.
    // Using the default constructor, the value will usually be in the range
    // [0-1], but can exceed 1 occasionally.
    float get_value(float timestamp);

    // Returns a neuron weight corresponding to the cumulative input.
    // Guaranteed to be in the range [0, 127]
    int8_t get_weight(float timestamp);

    // Applies a spike, increasing the value.
    void spike(float timestamp);

    // Applies a 'negative' spike, decreasing the value.
    // This should reverse the effect of a call to spike().
    void negative_spike(float timestamp);

    // Resets the decay timer and sets the value to zero.
    void reset();

  private:
    // The current value.
    float value;

    // How much to increase the value when a spike is received.
    // The distance between the value and 1 is closed by this fraction.
    const float spike_fraction;

    // Utility class to calculate decay efficiently.
    DecayCalculator decay_calculator;

    // Decays the value to the specified time.
    void decay_value(float timestamp);
};

#endif // _decaying_value_h
