#include "spike_scheduler.h"
#include "token.h"
#include "token_output.h"

// Decodes the scheduled spikes.
// Returns false if the spikes can't be decoded.
static bool decode_spikes(
    SpikeScheduler* spike_scheduler,
    TokenOutput* token_output
) {
  token_output->reset();
  for (;;) {
    const ScheduledSpike* scheduled_spike = spike_scheduler->peek_next();
    if (scheduled_spike == nullptr) {
      break;
    }
    token_output->spike({scheduled_spike->channel});
    spike_scheduler->advance();
  }

  // Exit if no token passes the validity threshold.
  const Token* best_token = token_output->best_token();
  if (best_token == nullptr) {
    printf("\n");
    return false;
  }
  // Print the token.
  if (best_token->is_suffix) {
    printf("%s", best_token->text.c_str());
  } else {
    printf(" %s", best_token->text.c_str());
  }
  fflush(stdout);
  return true;
}

// Reads the tokens from the specified file and transcodes them.
// Returns false if there's a problem.
static bool transcode_tokens(
    const Parameters& parameters,
    const char* path,
    const std::vector<Token>& tokens
) {
  FILE* fp;
  if ((fp = fopen(path, "r")) == nullptr) {
    fprintf(stderr, "fopen %s: %m\n", path);
    return false;
  }

  SpikeScheduler spike_scheduler(tokens[0].num_channels, parameters);
  TokenOutput token_output;
  token_output.set_tokens(tokens);
  const float duration = parameters.SECONDS_PER_SAMPLE;
  float timestamp = 0;
  for (;;) {
    uint16_t token_id;
    if (fread(&token_id, sizeof(uint16_t), 1, fp) != 1) {
      break;
    }
    spike_scheduler.schedule_embedding(
        timestamp, duration, tokens[token_id].embedding, /* randomize= */ true);
    timestamp += duration;
    if (!decode_spikes(&spike_scheduler, &token_output)) {
      break;
    }
  }
  printf("\n");

  fclose(fp);
  return true;
}

int main(int argc, char** argv) {
  const Parameters parameters = Parameters::DEFAULT_PARAMETERS;

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

  if (!transcode_tokens(parameters, "data/economist.tok", tokens)) {
    return 1;
  }
  return 0;
}
