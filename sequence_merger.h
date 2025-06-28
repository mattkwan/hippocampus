#ifndef _sequence_merger_h
#define _sequence_merger_h

#include "spike_queue.h"
#include "spike_scheduler.h"

// Merges sequences of spikes from a SpikeScheduler and an optional SpikeQueue
// to provide a time-ordered sequence of spikes up to a deadline.
class SequenceMerger {
  public:
    // Constructor.
    // The output queue is optional, and can be null. If it isn't null, it's
    // mutable, allowing spikes to be added at any time.
    // Spikes will be provided until the deadline.
    SequenceMerger(
        SpikeScheduler* spike_scheduler,
        SpikeQueue* output_queue,
        float deadline);

    // If there is a spike before the deadline, returns true, sets its
    // timestamp and deadline, and advances to the next spike.
    bool get_next(float* timestamp, uint16_t* channel);

  private:
    // Provides a time-ordered sequence of spikes.
    SpikeScheduler* spike_scheduler;

    // Nullable. Provides a time-ordered sequence of spikes. Can be appended to
    // at any time.
    SpikeQueue* spike_queue;

    // Spikes will only be provided if their timestamp is before this deadline.
    const float deadline;
};

#endif // _sequence_merger_h
