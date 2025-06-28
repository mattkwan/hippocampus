#ifndef _token_output_h
#define _token_output_h

#include "output_state.h"

#include <vector>

// Maintains the state of the output tokens.
class TokenOutput {
  public:
    // Sets the tokens used for output. The tokens should persist.
    // This should be called once.
    void set_tokens(const std::vector<Token>& tokens);

    // Processes spikes on the specified channels.
    // Uses the values to activate output tokens.
    void spike(const std::vector<uint16_t>& channels);

    // Returns the token with the highest valid activation level.
    // Returns nullptr if none exceed a validity threshold.
    const Token* best_token();

    // Resets the output state.
    void reset();

  private:
    // The state of the output tokens.
    std::vector<OutputState> output_states;
};

#endif // _token_output_h
