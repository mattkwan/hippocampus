#include "spike_queue.h"

void SpikeQueue::add(const float timestamp, const uint16_t channel) {
  if (scheduled_spikes.empty()
      || timestamp >= scheduled_spikes.back().timestamp) {
    scheduled_spikes.push_back({timestamp, channel});
  } else if (timestamp < scheduled_spikes.front().timestamp) {
    scheduled_spikes.push_front({timestamp, channel});
  } else {
    for (auto it = scheduled_spikes.rbegin(); it != scheduled_spikes.rend();
        ++it) {
      if (timestamp >= (*it).timestamp) {
        scheduled_spikes.insert(it.base(), {timestamp, channel});
        break;
      }
    }
  }
}
