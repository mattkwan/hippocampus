#ifndef _output_state_h
#define _output_state_h

#include "token.h"

// Maintains the state of an output token.
class OutputState {
  public:
    // Constructor. The token must be persistent.
    OutputState(const Token& token);

    // Copy constructor.
    OutputState(const OutputState& output_state);

    // Destructor.
    ~OutputState();

    // Processes a spike on the specified channel.
    void spike(const uint16_t channel) {
      activation_level += embedding_weights[channel];
    }

    // Returns the output's activation level.
    float get_activation_level() const { return activation_level; }

    // Returns a reference to the token.
    const Token& get_token() const { return token; }

    // Resets the output state.
    void reset();

  private:
    // A reference to the token.
    const Token& token;

    // The activation level of the output token.
    float activation_level;

    // The token's embedding, normalized to [0, 1).
    float* const embedding_weights;
};

#endif // _output_state_h
