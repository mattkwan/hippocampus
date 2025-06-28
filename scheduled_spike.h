#ifndef _scheduled_spike_h
#define _scheduled_spike_h

#include <cstdint>

// Details of a scheduled spike.
// Ideally the fields would be const, but there's no way to initialize an array
// of structs.
struct ScheduledSpike {
  float timestamp;
  uint16_t channel;
};

#endif // _scheduled_spike_h
