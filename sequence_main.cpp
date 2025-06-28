#include "brain.h"
#include "sequence_merger.h"
#include "spike_scheduler.h"

#include <cstdio>
#include <cstdlib>

// Schedules spikes to create a sequence of unique vectors.
static void schedule_training_spikes(
    const std::vector<float>& pattern,
    const unsigned int sequence_length,
    const Parameters& parameters,
    SpikeScheduler* spike_scheduler
) {
  // Schedule a sequence of vectors.
  for (unsigned int i = 0; i < sequence_length; i++) {
    const float start_time = i * parameters.SECONDS_PER_SAMPLE;
    unsigned int channel = i * pattern.size();
    for (const float value : pattern) {
      spike_scheduler->schedule_value(
          start_time,
          parameters.SECONDS_PER_SAMPLE,
          channel,
          value,
          /* randomize= */ true);
      channel++;
    }
  }
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

// Trains a brain using a sequence of vectors.
static void train_brain_sequence(
    const uint16_t num_channels,
    const std::vector<float>& pattern,
    const unsigned int sequence_length,
    const Parameters& parameters,
    Brain* brain
) {
  SpikeScheduler spike_scheduler(num_channels, parameters);
  schedule_training_spikes(
      pattern, sequence_length, parameters, &spike_scheduler);
  apply_training_spikes(parameters, &spike_scheduler, brain);
}

// Schedules spikes to create the initial vector.
static void schedule_testing_spikes(
    const std::vector<float>& pattern,
    const Parameters& parameters,
    SpikeScheduler* spike_scheduler
) {
  unsigned int channel = 0;
  for (const float value : pattern) {
    spike_scheduler->schedule_value(
        0,
        parameters.SECONDS_PER_SAMPLE,
        channel,
        value,
        /* randomize= */ true);
    channel++;
  }
}

// Reports on the values produced by the cortex.
// Includes an indication of which pattern has the strongest activation.
static void report_values(
  uint16_t num_channels,
  unsigned int* values,
  const float timestamp,
  const std::vector<float>& pattern,
  const unsigned int sequence_length,
  const Parameters& parameters
) {
  const unsigned int pstep = pattern.size();
  float pattern_activations[sequence_length] = {0};

  printf("%4.2f:", timestamp);
  for (uint16_t i = 0; i < num_channels; i++) {
    const float value = values[i];
    printf(" %2u", values[i]);
    values[i] = 0;

    for (unsigned int pid = 0; pid < sequence_length; pid++) {
      const unsigned int poffset = pid * pstep;
      if (i >= poffset && i < poffset + pstep) {
        pattern_activations[pid] += value * pattern[i - poffset];
      }
    }
  }
  printf("\n");

  unsigned int hi_idx = 0;
  float hi_value = 0;
  for (unsigned int i = 0; i < sequence_length; i++) {
    if (pattern_activations[i] > hi_value) {
      hi_value = pattern_activations[i];
      hi_idx = i;
    }
  }

  // Highlight the highest activation pattern.
  printf("      ");
  for (unsigned int i = 0; i < hi_idx * pstep; i++) {
    printf("   ");
  }
  printf("^^\n");
}

// Returns a random feedback delay.
// This is to prevent neurons from firing simultaneously, which can cause
// problems.
static float random_feedback_delay(const Parameters& parameters) {
  return parameters.MIN_SPIKE_INTERVAL * (1.0f + 2.0 * drand48());
}

// Provides spikes to the brain and reports how the generated sequence
// progresses.
static void apply_testing_spikes(
    const uint16_t num_channels,
    const std::vector<float>& pattern,
    const unsigned int sequence_length,
    const Parameters& parameters,
    SpikeScheduler* spike_scheduler,
    Brain* brain
) {
  // Apply the prompt spikes and keep feeding back the cortex output and
  // printing the output.
  const float duration = parameters.SECONDS_PER_SAMPLE * (sequence_length + 2);
  SpikeQueue feedback_queue;
  SequenceMerger sequence_merger(spike_scheduler, &feedback_queue, duration);
  std::vector<uint16_t> output_spikes;

  // Outputs per channel.
  unsigned int values[num_channels] = {0};

  const float reporting_interval = 0.1f;
  float reporting_deadline = reporting_interval;
  float timestamp;
  uint16_t channel;
  while (sequence_merger.get_next(&timestamp, &channel)) {
    brain->spike(
        timestamp,
        channel,
        /* use_hippocampus= */ false,
        parameters,
        &output_spikes);
    spike_scheduler->advance();

    for (uint16_t chan : output_spikes) {
      const float feedback_delay = random_feedback_delay(parameters);
      feedback_queue.add(timestamp + feedback_delay, chan);
      values[chan]++;
    }
    output_spikes.clear();

    if (timestamp >= reporting_deadline) {
      report_values(
          num_channels,
          values,
          reporting_deadline,
          pattern,
          sequence_length,
          parameters);
      reporting_deadline += reporting_interval;
    }
  }
  report_values(
      num_channels,
      values,
      reporting_deadline,
      pattern,
      sequence_length,
      parameters);
}

// Provides spikes to the brain from the start of the sequence and prints how
// the generated sequence progresses.
static void test_brain_sequence(
    const uint16_t num_channels,
    const std::vector<float> pattern,
    const unsigned int sequence_length,
    const Parameters& parameters,
    Brain* brain
) {
  SpikeScheduler spike_scheduler(num_channels, parameters);
  schedule_testing_spikes(pattern, parameters, &spike_scheduler);
  apply_testing_spikes(
      num_channels,
      pattern,
      sequence_length,
      parameters,
      &spike_scheduler,
      brain);
}

int main(int argc, char** argv) {
  const Parameters parameters(
      /* MIN_SPIKE_INTERVAL= */ 0.01f,
      /* SECONDS_PER_SAMPLE= */ 0.5f,
      /* SPIKE_FRACTION= */ 0.08f,
      /* DECAY_HALF_LIFE= */ 0.5f,
      /* NEGATIVE_SPIKE_FRACTION= */ 0.08f,
      /* NEGATIVE_WEIGHT_HALF_LIFE= */ 5.0f);

  const std::vector<float> pattern = {0.51f, 0.51f};
  const unsigned int sequence_length = 8;
  const uint16_t num_channels = pattern.size() * sequence_length;

  Brain brain(num_channels, parameters);
  brain.reserve(num_channels * 100);

  train_brain_sequence(
      num_channels, pattern, sequence_length, parameters, &brain);

  printf("%u neurons created during training.\n", brain.neuron_count());
  brain.reset();

  test_brain_sequence(
      num_channels, pattern, sequence_length, parameters, &brain);

  return 0;
}
