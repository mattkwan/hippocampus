#ifndef _spike_queue_h
#define _spike_queue_h

#include "scheduled_spike.h"

#include <deque>

// Maintains a queue of scheduled spikes, ordered by scheduled time.
class SpikeQueue {
  public:
    // Schedules a spike at the specified time.
    void add(float timestamp, uint16_t channel);

    // Returns true if there are no scheduled spikes.
    bool empty() const { return scheduled_spikes.empty(); }

    // Returns a reference to the next scheduled spike.
    const ScheduledSpike& front() const { return scheduled_spikes.front(); }

    // Removes the first scheduled spike.
    void pop() { scheduled_spikes.pop_front(); }

  private:
    // The scheduled spikes.
    std::deque<ScheduledSpike> scheduled_spikes;
};

#endif // _spike_queue_h
