#include "token_output.h"

void TokenOutput::set_tokens(const std::vector<Token>& tokens) {
  output_states.clear();
  if (tokens.empty()) {
    return;
  }
  output_states.reserve(tokens.size());
  for (const Token& token : tokens) {
    output_states.emplace_back(token);
  }
}

void TokenOutput::spike(const std::vector<uint16_t>& channels) {
  for (const uint16_t channel : channels) {
    for (OutputState& output_state : output_states) {
      output_state.spike(channel);
    }
  }
}

void TokenOutput::reset() {
  for (OutputState& output_state : output_states) {
    output_state.reset();
  }
}

const Token* TokenOutput::best_token() {
  float best_activation_level = 0;
  const Token* best_token = nullptr;

  for (OutputState& output_state : output_states) {
    const float activation_level = output_state.get_activation_level();
    if (activation_level > best_activation_level) {
      best_activation_level = activation_level;
      best_token = &output_state.get_token();
    }
  }
  return best_token;
}
