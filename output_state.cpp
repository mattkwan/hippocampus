#include "output_state.h"

OutputState::OutputState(const Token& token_) :
  token(token_),
  activation_level(0),
  embedding_weights(new float[token.num_channels])
{
  for (uint16_t i = 0; i < token.num_channels; i++) {
    embedding_weights[i] = token_.embedding[i] / 256.0f;
  }
}

OutputState::OutputState(const OutputState& output_state) :
  token(output_state.token),
  activation_level(0),
  embedding_weights(new float[token.num_channels])
{
  for (uint16_t i = 0; i < token.num_channels; i++) {
    embedding_weights[i] = output_state.embedding_weights[i];
  }
}

OutputState::~OutputState() {
  delete[] embedding_weights;
}

void OutputState::reset() {
  activation_level = 0;
}
