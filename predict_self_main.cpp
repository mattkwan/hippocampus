#include "brain.h"
#include "spike_scheduler.h"
#include "token.h"
#include "token_output.h"

#include <cmath>
#include <ctime>
#include <getopt.h>

// Prints the token decoded from spikes.
// Returns false if the spikes can't be decoded.
static bool print_token_output(
    const unsigned int idx,
    const unsigned int neuron_count,
    const float correlation,
    const float relative_volume,
    TokenOutput* token_output
) {
  printf("%u n=%u corr=%.3f vol=%.2f ",
      idx, neuron_count, correlation, relative_volume);

  // Exit if no token passes the validity threshold.
  const Token* best_token = token_output->best_token();
  if (best_token == nullptr) {
    printf("no output\n");
    return false;
  }
  if (best_token->is_suffix) {
    printf("-");
  }
  printf("%s\n", best_token->text.c_str());
  return true;
}

// Applies the scheduled spikes to the brain and prints its output.
// Returns true if the output could be decoded.
static void apply_spikes_to_brain(
    const Parameters& parameters,
    const bool use_hippocampus,
    SpikeScheduler* spike_scheduler,
    Brain* brain,
    unsigned int* inputs_count,
    unsigned int* outputs_count,
    TokenOutput* token_output
) {
  std::vector<uint16_t> outputs;
  for (;;) {
    const ScheduledSpike* scheduled_spike = spike_scheduler->peek_next();
    if (scheduled_spike == nullptr) {
      break;
    }
    inputs_count[scheduled_spike->channel]++;
    brain->spike(
        scheduled_spike->timestamp,
        scheduled_spike->channel,
        use_hippocampus,
        parameters,
        &outputs);
    if (token_output != nullptr) {
      token_output->spike(outputs);
    }
    for (const uint16_t channel : outputs) {
      outputs_count[channel]++;
    }
    outputs.clear();
    spike_scheduler->advance();
  }
}

// Evaluates the output of the brain when it is fed noise, i.e. an average
// signal on all channels.
// If the neuron weights have been trained correctly there should be no output.
static void evaluate_noise(
    const uint16_t num_channels,
    const Parameters& parameters,
    const bool randomize,
    Brain* brain
) {
  uint8_t embedding[num_channels];
  for (unsigned int i = 0; i < num_channels; i++) {
    embedding[i] = 128;
  }

  SpikeScheduler spike_scheduler(num_channels, parameters);
  spike_scheduler.schedule_embedding(
      /* timestamp= */ 0,
      /* duration= */ parameters.SECONDS_PER_SAMPLE,
      embedding,
      randomize);

  unsigned int inputs_count[num_channels] = {0};
  unsigned int outputs_count[num_channels] = {0};
  brain->reset();
  apply_spikes_to_brain(
      parameters,
      /* use_hippocampus= */ false,
      &spike_scheduler,
      brain,
      inputs_count,
      outputs_count,
      /* token_output= */ nullptr);

  unsigned int in_sum = 0;
  unsigned int out_sum = 0;
  for (unsigned int i = 0; i < num_channels; i++) {
    in_sum += inputs_count[i];
    out_sum += outputs_count[i];
  }
  printf("Noise in/out %u/%u\n", in_sum, out_sum);
}

// Returns the weighted ratio of output spikes to input spikes.
static float compare_inputs_outputs(
    const unsigned int* inputs_count,
    const unsigned int* outputs_count,
    const unsigned int n
) {
  float weighted_inputs = 0;
  float weighted_outputs = 0;
  for (unsigned int i = 0; i < n; i++) {
    const unsigned int ival = inputs_count[i];
    const unsigned int oval = outputs_count[i];
    weighted_inputs += ival * ival;
    weighted_outputs += ival * oval;
  }
  return weighted_inputs == 0 ? 0 : weighted_outputs / weighted_inputs;
}

// Returns the correlation coefficient between the inputs and outputs.
static float correlation_coefficient(
    const uint8_t* inputs,
    const unsigned int* outputs,
    const unsigned int n
) {
  float sumx = 0;
  float sumy = 0;
  float sumxx = 0;
  float sumyy = 0;
  float sumxy = 0;
  for (unsigned int i = 0; i < n; i++) {
    const float x = inputs[i];
    const float y = outputs[i];
    sumx += x;
    sumy += y;
    sumxx += x * x;
    sumyy += y * y;
    sumxy += x * y;
  }
  const float d = sqrtf((n * sumxx - sumx * sumx) * (n * sumyy - sumy * sumy));
  return d == 0 ? 0 : (n * sumxy - sumx * sumy) / d;
}

// Repeatedly applies the token to a brain and prints the brain's output.
static void repeat_token(
    const Parameters& parameters,
    const uint16_t token_id,
    const unsigned int repeat_count,
    const bool randomize,
    const std::vector<Token>& tokens
) {
  const uint16_t num_channels = tokens[token_id].num_channels;
  Brain brain(num_channels, parameters);
  brain.reserve(num_channels * 100);

  SpikeScheduler spike_scheduler(num_channels, parameters);
  TokenOutput token_output;
  token_output.set_tokens(tokens);
  const float duration = parameters.SECONDS_PER_SAMPLE;
  float timestamp = 0;
  for (unsigned int i = 0; i < repeat_count; i++) {
    unsigned int inputs_count[num_channels] = {0};
    unsigned int outputs_count[num_channels] = {0};
    token_output.reset();
    spike_scheduler.schedule_embedding(
        timestamp,
        duration,
        tokens[token_id].embedding,
        randomize);
    apply_spikes_to_brain(
        parameters,
        /* use_hippocampus= */ true,
        &spike_scheduler,
        &brain,
        inputs_count,
        outputs_count,
        &token_output);
    const float correlation = correlation_coefficient(
        tokens[token_id].embedding,
        outputs_count,
        num_channels);
    const float relative_volume = compare_inputs_outputs(
        inputs_count, outputs_count, num_channels);
    timestamp += duration;
    print_token_output(
        i, brain.neuron_count(), correlation, relative_volume, &token_output);
  }

  evaluate_noise(num_channels, parameters, randomize, &brain);
}

// Returns the ID of the token that will be used to train the brain.
static uint16_t select_token_id(
    const std::vector<Token>& tokens,
    const bool randomize
) {
  if (!randomize) {
    // Token for "American".
    return 1102;
  }

  // Select a token at random.
  srand48(time(nullptr));
  const uint16_t token_id = lrand48() % tokens.size();
  const Token& token = tokens[token_id];

  printf("Random token (%u): ", token_id);
  if (token.is_suffix) {
    printf("-");
  }
  printf("%s\n", token.text.c_str());

  return token_id;
}

int main(int argc, char** argv) {
  int opt;
  bool randomize = false;
  while ((opt = getopt(argc, argv, "R")) != -1) {
    switch (opt) {
      case 'R':
        randomize = true;
        break;
      default:
        printf("Usage: %s [-R]\n", argv[0]);
        return 1;
    }
  }

  const Parameters parameters(
      /* MIN_SPIKE_INTERVAL= */ 0.01f,
      /* SECONDS_PER_SAMPLE= */ 0.2f,
      /* SPIKE_FRACTION= */ 0.08f,
      /* DECAY_HALF_LIFE= */ 0.5f,
      /* NEGATIVE_SPIKE_FRACTION= */ 0.08f,
      /* NEGATIVE_WEIGHT_HALF_LIFE= */ 5.0f);

  std::vector<Token> tokens;
  if (!Token::parse(
      "data/tokens-20k.raw",
      "data/embeddings-500.raw",
      500,
      &tokens)) {
    return 1;
  }
  printf("Parsed %lu tokens\n", tokens.size());
  if (tokens.empty()) {
    return 1;
  }

  const uint16_t token_id = select_token_id(tokens, randomize);
  repeat_token(parameters, token_id, 20, randomize, tokens);

  return 0;
}
