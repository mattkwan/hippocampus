#ifndef _decay_calculator_h
#define _decay_calculator_h

#include <cstdint>
#include <unordered_map>

// Utility class for calculating exponential decay efficiently.
class DecayCalculator {
  public:
    // Constructor.
    // The rate of decay is such that the decay after t seconds equals
    // e(t * decay_rate)
    DecayCalculator(float decay_rate);

    // Calculates the decay factor at the specified timestamp.
    // Returns true if the factor is low enough to be worth using.
    bool calculate_factor(float timestamp, float* factor);

    // Resets the decay timer.
    void reset();

  private:
    // The decay rate such that the decay after t seconds equals
    // e(-t * decay_rate)
    const float decay_rate;

    // The duration in seconds at which decay becomes meaningful.
    const float minimum_duration;

    // The number of pre-calculated decay factors.
    static constexpr int PRECALCULATED_FACTOR_COUNT = 1024;

    // Pre-calculated decay factors.
    const float* precalculated_factors;

    // The last time a useful decay factor was returned.
    float previous_timestamp;

    // A map of pre-calculated minimum durations for different decay rates.
    static std::unordered_map<float, float> minimum_duration_map;

    // A map of pre-calculated decay factors for different decay rates.
    static std::unordered_map<float, const float*> precalculated_factors_map;

    // Calculates the duration, in seconds, at which decay becomes meaningful.
    // Uses cached values wherever possible.
    static float calculate_minimum_duration(float decay_rate);

    // Calculates precalculated decay factors for a decay rate.
    // Uses cached values wherever possible.
    static const float* calculate_precalculated_factors(
        float mimimum_duration,
        float decay_rate);
};

#endif // _decay_calculator_h
