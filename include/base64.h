#ifndef BASE64_H
#define BASE64_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Base64 {

// Encode binary data to Base64 string
std::string encode(const uint8_t* data, size_t length);
std::string encode(const std::vector<uint8_t>& data);

// Decode Base64 string to binary data
// Returns empty vector on invalid input
std::vector<uint8_t> decode(const std::string& encoded);

// Check if string is valid Base64
bool isValid(const std::string& encoded);

// Calculate encoded size for given input length
size_t encodedSize(size_t inputLength);

// Calculate decoded size for given encoded string
// Returns 0 if string is invalid
size_t decodedSize(const std::string& encoded);

}  // namespace Base64

#endif  // BASE64_H
