#include "sequence_merger.h"

SequenceMerger::SequenceMerger(
    SpikeScheduler* spike_scheduler_,
    SpikeQueue* spike_queue_,
    const float deadline_
) :
  spike_scheduler(spike_scheduler_),
  spike_queue(spike_queue_),
  deadline(deadline_)
{
}

bool SequenceMerger::get_next(float* timestamp, uint16_t* channel) {
  // Check if there's anything valid in the spike scheduler.
  bool spike_scheduler_found = false;
  const ScheduledSpike* scheduled_spike = spike_scheduler->peek_next();
  if (scheduled_spike != nullptr && scheduled_spike->timestamp < deadline) {
    spike_scheduler_found = true;
  }

  // Check if there's anything valid in the spike queue.
  bool spike_queue_found = spike_queue != nullptr && !spike_queue->empty()
      && spike_queue->front().timestamp < deadline;

  // Return the spike with the earliest timestamp.
  if (spike_scheduler_found && (!spike_queue_found
      || scheduled_spike->timestamp <= spike_queue->front().timestamp)) {
    *timestamp = scheduled_spike->timestamp;
    *channel = scheduled_spike->channel;
    spike_scheduler->advance();
    return true;
  }
  if (spike_queue_found) {
    *timestamp = spike_queue->front().timestamp;
    *channel = spike_queue->front().channel;
    spike_queue->pop();
    return true;
  }
  return false;
}
