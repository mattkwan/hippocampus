#include "token.h"

#include <cstring>

Token::Token(
    const uint16_t id_,
    const bool is_suffix_,
    const std::string& text_,
    const uint8_t* embedding_,
    const uint16_t num_channels_
) :
  id(id_),
  is_suffix(is_suffix_),
  text(text_),
  num_channels(num_channels_),
  embedding(new uint8_t[num_channels])
{
  memcpy(embedding, embedding_, num_channels * sizeof(uint8_t));
}

Token::Token(const Token& token) :
  id(token.id),
  is_suffix(token.is_suffix),
  text(token.text),
  num_channels(token.num_channels),
  embedding(new uint8_t[token.num_channels])
{
  memcpy(embedding, token.embedding, token.num_channels * sizeof(uint8_t));
}

Token::~Token() {
  delete[] embedding;
}

// Parses the next token from the two files and appends it to the list.
// Returns false if it can't read the next token.
static bool parse_token(
    const uint16_t token_id,
    const uint16_t num_channels,
    FILE* sfp,
    FILE* efp,
    std::vector<Token>* tokens
) {
  // Parse the embedding values.
  uint8_t embedding[num_channels];
  if (fread(embedding, sizeof(uint8_t), num_channels, efp) != num_channels) {
    fprintf(stderr, "fread embedding: %m\n");
    return false;
  }

  // Parse whether the token is a suffix.
  int is_suffix = fgetc(sfp);
  if (is_suffix < 0) {
    fprintf(stderr, "fread suffix flag: %m\n");
    return false;
  } else if (is_suffix != 0 && is_suffix != 1) {
    fprintf(stderr, "invalid suffix flag: %d\n", is_suffix);
    return false;
  }

  // Parse the token text.
  int len = fgetc(sfp);
  if (len < 0) {
    fprintf(stderr, "fread string length: %m\n");
    return false;
  }
  char text[len + 1];
  if (fread(text, 1, len, sfp) != (unsigned int) len) {
    fprintf(stderr, "fread string %d: %m\n", len);
    return false;
  }
  text[len] = '\0';

  tokens->emplace_back(
      token_id,
      is_suffix == 1,
      (char *) text,
      (const uint8_t*) embedding,
      num_channels);
  return true;
}

// Parses a list of tokens from the two files.
static bool parse_files(
    const uint16_t num_channels,
    FILE* sfp,
    FILE* efp,
    std::vector<Token>* tokens
) {
  // Parse the number of tokens.
  uint16_t num_tokens;
  if (fread(&num_tokens, sizeof(uint16_t), 1, sfp) != 1) {
    return false;
  }

  // Parse the actual tokens.
  for (uint16_t token_id = 0; token_id < num_tokens; token_id++) {
    if (!parse_token(token_id, num_channels, sfp, efp, tokens)) {
      return false;
    }
  }
  return true;
}

bool Token::parse(
    const char* strings_path,
    const char* embeddings_path,
    const uint16_t num_channels,
    std::vector<Token>* tokens
) {
  FILE* sfp;
  FILE* efp;

  if ((sfp = fopen(strings_path, "r")) == nullptr) {
    fprintf(stderr, "fopen %s: %m\n", strings_path);
    return false;
  }
  if ((efp = fopen(embeddings_path, "r")) == nullptr) {
    fprintf(stderr, "fopen %s: %m\n", embeddings_path);
    fclose(sfp);
    return false;
  }

  const bool ok = parse_files(num_channels, sfp, efp, tokens);
  fclose(efp);
  fclose(sfp);
  return ok;
}
