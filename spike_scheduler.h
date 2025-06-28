#ifndef _spike_scheduler_h
#define _spike_scheduler_h

#include "parameters.h"
#include "scheduled_spike.h"

// A scheduler for spikes.
class SpikeScheduler {
  public:
    // Constructor.
    SpikeScheduler(uint16_t num_channels, const Parameters& parameters);

    // Disable the copy constructor.
    SpikeScheduler(const SpikeScheduler& spike_scheduler) = delete;

    // Destructor.
    ~SpikeScheduler();

    // Converts a [0-1] value into time-ordered spikes on a channel.
    // The start timestamp and duration are both expressed in seconds.
    void schedule_value(
        float start_timestamp,
        float duration,
        uint16_t channel,
        float value,
        bool randomize);

    // Converts an embedding into a time-ordered sequence of spikes.
    // The start timestamp and duration are both expressed in seconds.
    void schedule_embedding(
        float start_timestamp,
        float duration,
        const uint8_t* embedding,
        bool randomize);

    // Returns a pointer to the next scheduled spike, or null if the are none.
    const ScheduledSpike* peek_next() const;

    // Advances to the next scheduled spike.
    void advance() { next_scheduled_spike++; }

  private:
    // The number of channels.
    const uint16_t num_channels;

    // The shortest interval between spikes.
    const float min_spike_interval;

    // The incremental value of a spike, somewhere between zero and one.
    // The value will be incremented by this fraction of the distance to one.
    const float spike_fraction;

    // The number of scheduled spikes.
    unsigned int num_spikes;

    // The index of the next scheduled spike.
    unsigned int next_scheduled_spike;

    // The maximum number of spikes that can be stored in the allocated array.
    unsigned int allocated_spikes;

    // The scheduled spikes.
    ScheduledSpike* scheduled_spikes;

    // Whether the random number generator has been seeded.
    static bool is_random_seeded;

    // Allocates enough space for the specified number of additional spikes.
    // Also deletes any previously-consumed spikes.
    void allocate_additional_spikes(unsigned int n);

    // Calculates and returns the interval between spikes to represent the
    // value. Returns zero if the value can't be represented.
    float calculate_period(float value) const;

    // Seeds the random number generator.
    static void seed_random_generator();
};

#endif // _spike_scheduler_h
