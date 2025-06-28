#ifndef _token_h
#define _token_h

#include <cstdint>
#include <string>
#include <vector>

// A token, representing a string and an embedding.
class Token {
  public:
    // Constructor.
    Token(
        uint16_t id,
        bool is_suffix,
        const std::string& text,
        const uint8_t* embedding,
        const uint16_t num_channels);

    // Copy constructor.
    Token(const Token& token);

    // Destructor.
    ~Token();

    // The token's ID.
    const uint16_t id;

    // Whether the token is a suffix (to be appended to the previous token),
    // or a new word.
    const bool is_suffix;

    // The text string associated with the token.
    const std::string text;

    // The number of values in the embedding.
    const uint16_t num_channels;

    // The token's embedding.
    uint8_t* const embedding;

    // Parses the token strings and token embeddings files, returning a list
    // of tokens.
    // Returns true if both files are successfully parsed.
    static bool parse(
        const char* strings_path,
        const char* embeddings_path,
        const uint16_t num_channels,
        std::vector<Token>* tokens);
};

#endif // _token_h
