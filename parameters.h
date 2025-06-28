#ifndef _parameters_h
#define _parameters_h

// Various parameters that control the system.
class Parameters {
  public:
    // Constructor.
    Parameters(
        float min_spike_interval,
        float seconds_per_sample,
        float spike_fraction,
        float decay_half_life,
        float negative_spike_fraction,
        float negative_weight_half_life);

    // The minimum interval between spikes, in seconds.
    const float MIN_SPIKE_INTERVAL;

    // The duration that spikes will be applied for each embedding.
    const float SECONDS_PER_SAMPLE;

    // The fraction of the distance to 1 added by a spike.
    const float SPIKE_FRACTION;

    // The exponential decay half life (in seconds) applied to spikes.
    const float DECAY_HALF_LIFE;

    // The fraction of the distance to 1 added by a spike to negative weights.
    const float NEGATIVE_SPIKE_FRACTION;

    // The exponential decay half life applied to negative weights.
    const float NEGATIVE_WEIGHT_HALF_LIFE;

    // Static parameters containing default values.
    static const Parameters DEFAULT_PARAMETERS;
};

#endif // _parameters_h
