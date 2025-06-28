#include "spike_scheduler.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>

bool SpikeScheduler::is_random_seeded = false;

SpikeScheduler::SpikeScheduler(
    const uint16_t num_channels_,
    const Parameters& parameters
) :
  num_channels(num_channels_),
  min_spike_interval(parameters.MIN_SPIKE_INTERVAL),
  spike_fraction(parameters.SPIKE_FRACTION),
  num_spikes(0),
  next_scheduled_spike(0),
  allocated_spikes(0),
  scheduled_spikes(nullptr)
{
}

SpikeScheduler::~SpikeScheduler() {
  if (scheduled_spikes != nullptr) {
    delete[] scheduled_spikes;
  }
}

void SpikeScheduler::seed_random_generator() {
  if (is_random_seeded) {
    return;
  }
  srand48(time(nullptr));
  is_random_seeded = true;
}

float SpikeScheduler::calculate_period(const float value) const {
  if (value <= spike_fraction) {
    return 0;
  } else if (value > 1) {
    return min_spike_interval;
  } else {
    return min_spike_interval / value;
  }
}

// Returns the number of spikes when they occur at the specified frequency (in
// Hertz), for the specified duration (in seconds).
static unsigned int calculate_spike_count(
    const float period,
    const float duration
) {
  return 1 + (unsigned int) floorf(duration / period);
}

// Compares two scheduled spikes by ascending timestamp.
static int scheduled_spike_cmp(const void* p1, const void* p2) {
  const ScheduledSpike* ssp1 = (const ScheduledSpike*) p1;
  const ScheduledSpike* ssp2 = (const ScheduledSpike*) p2;
  const float diff = ssp1->timestamp - ssp2->timestamp;
  if (diff < 0) {
    return -1;
  } else if (diff > 0) {
    return 1;
  } else {
    return 0;
  }
}

void SpikeScheduler::schedule_value(
    const float start_timestamp,
    const float duration,
    const uint16_t channel,
    const float value,
    const bool randomize
) {
  if (randomize) {
    seed_random_generator();
  }

  // Calculate the period for encoding the value. Abort if the value can't be
  // encoded.
  const float period = calculate_period(value);
  if (period <= 0) {
    return;
  }

  // Calculate the time of the first spike. Abort if it doesn't occur during
  // the duration.
  const float start_offset_fraction = randomize ? (float) drand48() : 0.5f;
  const float start_offset = start_offset_fraction * period;
  if (start_offset > duration - min_spike_interval) {
    return;
  }

  // Calculate how many spikes will be added and allocate space for them.
  const unsigned int count = calculate_spike_count(
      period, duration - start_offset - min_spike_interval);
  allocate_additional_spikes(count);

  float timestamp = start_timestamp + start_offset;
  for (unsigned int i = 0; i < count; i++) {
    ScheduledSpike* scheduled_spike = &scheduled_spikes[num_spikes + i];
    scheduled_spike->timestamp = timestamp;
    scheduled_spike->channel = channel;
    timestamp += period;
  }

  // Sort the spikes by timestamp.
  num_spikes += count;
  qsort(
      scheduled_spikes,
      num_spikes,
      sizeof(ScheduledSpike),
      scheduled_spike_cmp);
}

void SpikeScheduler::schedule_embedding(
    const float start_timestamp,
    const float duration,
    const uint8_t* embedding,
    const bool randomize
) {
  if (randomize) {
    seed_random_generator();
  }

  // Pre-calculate the spike count, period, and offset for each channel.
  float periods[num_channels];
  float start_offsets[num_channels];
  unsigned int counts[num_channels];
  unsigned int total_count = 0;

  for (uint16_t i = 0; i < num_channels; i++) {
    // Calculate the period for encoding the channel value. Skip if the value
    // can't be encoded.
    periods[i] = calculate_period(embedding[i] / 256.0f);
    if (periods[i] == 0) {
      counts[i] = 0;
      start_offsets[i] = 0;
      continue;
    }

    // Calculate the time of the first spike. Skip if it doesn't occur during
    // the duration.
    const float start_offset_fraction = randomize ? (float) drand48() : 0.5f;
    start_offsets[i] = start_offset_fraction * periods[i];
    if (start_offsets[i] > duration - min_spike_interval) {
      counts[i] = 0;
      continue;
    }

    // Calculate how many spikes will be added.
    counts[i] = calculate_spike_count(
        periods[i], duration - start_offsets[i] - min_spike_interval);
    total_count += counts[i];
  }
  if (total_count == 0) {
    return;
  }
  allocate_additional_spikes(total_count);

  unsigned int ssidx = num_spikes;
  for (uint16_t i = 0; i < num_channels; i++) {
    const unsigned int count = counts[i];
    if (count == 0) {
      continue;
    }
    const float period = periods[i];
    float timestamp = start_timestamp + start_offsets[i];
    for (unsigned int j = 0; j < count; j++) {
      scheduled_spikes[ssidx].timestamp = timestamp;
      scheduled_spikes[ssidx].channel = i;
      timestamp += period;
      ssidx++;
    }
  }

  next_scheduled_spike = 0;
  num_spikes = ssidx;

  // Sort the spikes by timestamp.
  qsort(
      scheduled_spikes,
      num_spikes,
      sizeof(ScheduledSpike),
      scheduled_spike_cmp);
}

const ScheduledSpike* SpikeScheduler::peek_next() const {
  return next_scheduled_spike < num_spikes
      ? &scheduled_spikes[next_scheduled_spike] : nullptr;
}

// Returns a value greater than the requested size, and a power of two.
static unsigned int add_headroom(const unsigned int requested_size) {
  unsigned int n = 64;
  while (n <= requested_size) {
    n <<= 1;
  }
  return n;
}

void SpikeScheduler::allocate_additional_spikes(const unsigned int n) {
  if (next_scheduled_spike > 0) {
    // Delete consumed spikes.
    memmove(
        scheduled_spikes,
        scheduled_spikes + next_scheduled_spike, 
        (num_spikes - next_scheduled_spike) * sizeof(ScheduledSpike));
    num_spikes -= next_scheduled_spike;
    next_scheduled_spike = 0;
  }
  if (allocated_spikes >= num_spikes + n) {
    // Enough space is already available.
    return;
  }
  // Allocate a new array and copy the existing spikes across.
  const unsigned int new_size = add_headroom(num_spikes + n);
  ScheduledSpike* new_spikes = new ScheduledSpike[new_size];
  if (scheduled_spikes != nullptr) {
    if (num_spikes > 0) {
      memcpy(new_spikes, scheduled_spikes, num_spikes * sizeof(ScheduledSpike));
    }
    delete[] scheduled_spikes;
  }
  scheduled_spikes = new_spikes;
  allocated_spikes = new_size;
}
