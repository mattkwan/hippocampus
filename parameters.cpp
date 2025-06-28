#include "parameters.h"

const Parameters Parameters::DEFAULT_PARAMETERS(
    /* MIN_SPIKE_INTERVAL= */ 0.01f,
    /* SECONDS_PER_SAMPLE= */ 0.5f,
    /* SPIKE_FRACTION= */ 0.08f,
    /* DECAY_HALF_LIFE= */ 0.5f,
    /* NEGATIVE_SPIKE_FRACTION= */ 0.08f,
    /* NEGATIVE_WEIGHT_HALF_LIFE= */ 5.0f);

Parameters::Parameters(
    const float min_spike_interval,
    const float seconds_per_sample,
    const float spike_fraction,
    const float decay_half_life,
    const float negative_spike_fraction,
    const float negative_weight_half_life
) :
  MIN_SPIKE_INTERVAL(min_spike_interval),
  SECONDS_PER_SAMPLE(seconds_per_sample),
  SPIKE_FRACTION(spike_fraction),
  DECAY_HALF_LIFE(decay_half_life),
  NEGATIVE_SPIKE_FRACTION(negative_spike_fraction),
  NEGATIVE_WEIGHT_HALF_LIFE(negative_weight_half_life) {
}
