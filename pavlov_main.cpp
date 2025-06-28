#include "brain.h"
#include "parameters.h"
#include "spike_scheduler.h"

#include <cstdio>

// Schedules spikes to create a "bell" stimulus followed by a "food" stimulus.
static void schedule_training_spikes(
    const float bell_duration,
    const float bell_intensity,
    const float gap_duration,
    const float food_duration,
    const float food_intensity,
    const Parameters& parameters,
    SpikeScheduler* spike_scheduler
) {
  // Channels 0-2 are for the bell, 3-5 are for the food.
  spike_scheduler->schedule_value(0, bell_duration, 0, bell_intensity, true);
  spike_scheduler->schedule_value(0, bell_duration, 1, bell_intensity, true);
  spike_scheduler->schedule_value(0, bell_duration, 2, bell_intensity, true);

  const float food_start = bell_duration + gap_duration;
  spike_scheduler->schedule_value(
      food_start, food_duration, 3, food_intensity, /* randomize= */ true);
  spike_scheduler->schedule_value(
      food_start, food_duration, 4, food_intensity, /* randomize= */ true);
  spike_scheduler->schedule_value(
      food_start, food_duration, 5, food_intensity, /* randomize= */ true);
}

// Applies the spikes to the brain to train it.
static void apply_training_spikes(
    const Parameters& parameters,
    SpikeScheduler* spike_scheduler,
    Brain* brain
) {
  // Apply the scheduled spikes, one at a time.
  std::vector<uint16_t> outputs;
  for (;;) {
    const ScheduledSpike* scheduled_spike = spike_scheduler->peek_next();
    if (scheduled_spike == nullptr) {
      break;
    }
    brain->spike(
        scheduled_spike->timestamp,
        scheduled_spike->channel,
        /* use_hippocampus= */ true,
        parameters,
        &outputs);
    spike_scheduler->advance();
    // We don't care about the outputs during training.
    outputs.clear();
  }
}

// Trains the brain with a "bell" stimulus followed by a "food" stimulus.
static void train_brain_pavlovian(
    const uint16_t num_channels,
    const float bell_duration,
    const float bell_intensity,
    const float gap_duration,
    const float food_duration,
    const float food_intensity,
    const Parameters& parameters,
    Brain* brain
) {
  SpikeScheduler spike_scheduler(num_channels, parameters);
  schedule_training_spikes(
      bell_duration,
      bell_intensity,
      gap_duration,
      food_duration,
      food_intensity,
      parameters,
      &spike_scheduler);
  apply_training_spikes(parameters, &spike_scheduler, brain);
}

// Schedules spikes to create a "bell" stimulus.
static void schedule_testing_spikes(
    const float bell_duration,
    const float bell_intensity,
    const Parameters& parameters,
    SpikeScheduler* spike_scheduler
) {
  // Channels 0-2 are for the bell.
  spike_scheduler->schedule_value(0, bell_duration, 0, bell_intensity, true);
  spike_scheduler->schedule_value(0, bell_duration, 1, bell_intensity, true);
  spike_scheduler->schedule_value(0, bell_duration, 2, bell_intensity, true);
}

// Applies a "bell" stimulus to the brain and reports how it responds.
static void apply_testing_spikes(
    const uint16_t num_channels,
    const Parameters& parameters,
    SpikeScheduler* spike_scheduler,
    Brain* brain
) {
  // Apply the scheduled spikes, one at a time.
  std::vector<uint16_t> outputs;
  for (;;) {
    const ScheduledSpike* scheduled_spike = spike_scheduler->peek_next();
    if (scheduled_spike == nullptr) {
      break;
    }
    brain->spike(
        scheduled_spike->timestamp,
        scheduled_spike->channel,
        /* use_hippocampus= */ false,
        parameters,
        &outputs);
    spike_scheduler->advance();
  }

  // Analyze the outputs.
  unsigned int outputs_count[num_channels];
  for (uint16_t i = 0; i < num_channels; i++) {
    outputs_count[i] = 0;
  }
  for (const uint16_t channel : outputs) {
    outputs_count[channel]++;
  }
  printf("Output spikes with bell input. [0-2] bell, [3-5] food.\n");
  for (uint16_t i = 0; i < num_channels; i++) {
    printf("%u: %u\n", i, outputs_count[i]);
  }
}

// Tests how a trained brain responds to a "bell" stimulus.
static void test_brain_pavlovian(
    const uint16_t num_channels,
    const float bell_duration,
    const float bell_intensity,
    const Parameters& parameters,
    Brain* brain
) {
  SpikeScheduler spike_scheduler(num_channels, parameters);
  schedule_testing_spikes(
      bell_duration,
      bell_intensity,
      parameters,
      &spike_scheduler);
  apply_testing_spikes(num_channels, parameters, &spike_scheduler, brain);
}

// Tests that a brain can be trained with a "bell" stimulus followed by a
// "food" stimulus, and later recall "food" when a bell is presented.
static void test_pavlovian_learning(
    const uint16_t num_channels,
    const float bell_duration,
    const float bell_intensity,
    const float gap_duration,
    const float food_duration,
    const float food_intensity,
    const Parameters& parameters
) {
  Brain brain(num_channels, parameters);
  brain.reserve(num_channels * 100);

  train_brain_pavlovian(
      num_channels,
      bell_duration,
      bell_intensity,
      gap_duration,
      food_duration,
      food_intensity,
      parameters,
      &brain);

  printf("%u neurons created during training.\n", brain.neuron_count());
  brain.reset();

  test_brain_pavlovian(
      num_channels, bell_duration, bell_intensity, parameters, &brain);
}

int main(int argc, char** argv) {
  const Parameters parameters(
      /* MIN_SPIKE_INTERVAL= */ 0.01f,
      /* SECONDS_PER_SAMPLE= */ 0.5f,
      /* SPIKE_FRACTION= */ 0.08f,
      /* DECAY_HALF_LIFE= */ 0.5f,
      /* NEGATIVE_SPIKE_FRACTION= */ 0.08f,
      /* NEGATIVE_WEIGHT_HALF_LIFE= */ 5.0f);

  test_pavlovian_learning(
      /* num_channels= */ 6,
      /* bell_duration= */ 0.5f,
      /* bell_intensity= */ 0.7f,
      /* gap_duration= */ 0.25f,
      /* food_duration= */ 0.5f,
      /* food_intensity= */ 0.7f,
      parameters);

  return 0;
}
