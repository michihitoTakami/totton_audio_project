/**
 * @file test_base64.cpp
 * @brief Unit tests for Base64 encoding/decoding (RFC 4648).
 */

#include "base64.h"

#include <cstring>
#include <gtest/gtest.h>
#include <vector>

// ============================================================
// Encode Tests
// ============================================================

TEST(Base64, EncodeEmpty) {
    std::vector<uint8_t> empty;
    EXPECT_EQ(Base64::encode(empty), "");
}

TEST(Base64, EncodeOneByte) {
    std::vector<uint8_t> data = {0x4D};  // 'M'
    EXPECT_EQ(Base64::encode(data), "TQ==");
}

TEST(Base64, EncodeTwoBytes) {
    std::vector<uint8_t> data = {0x4D, 0x61};  // "Ma"
    EXPECT_EQ(Base64::encode(data), "TWE=");
}

TEST(Base64, EncodeThreeBytes) {
    std::vector<uint8_t> data = {0x4D, 0x61, 0x6E};  // "Man"
    EXPECT_EQ(Base64::encode(data), "TWFu");
}

TEST(Base64, EncodeHelloWorld) {
    const char* text = "Hello, World!";
    std::vector<uint8_t> data(text, text + strlen(text));
    EXPECT_EQ(Base64::encode(data), "SGVsbG8sIFdvcmxkIQ==");
}

TEST(Base64, EncodeBinaryData) {
    std::vector<uint8_t> data = {0x00, 0xFF, 0x7F, 0x80, 0x01};
    EXPECT_EQ(Base64::encode(data), "AP9/gAE=");
}

// ============================================================
// Decode Tests
// ============================================================

TEST(Base64, DecodeEmpty) {
    EXPECT_TRUE(Base64::decode("").empty());
}

TEST(Base64, DecodeOneByte) {
    auto result = Base64::decode("TQ==");
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 0x4D);
}

TEST(Base64, DecodeTwoBytes) {
    auto result = Base64::decode("TWE=");
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 0x4D);
    EXPECT_EQ(result[1], 0x61);
}

TEST(Base64, DecodeThreeBytes) {
    auto result = Base64::decode("TWFu");
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 0x4D);
    EXPECT_EQ(result[1], 0x61);
    EXPECT_EQ(result[2], 0x6E);
}

TEST(Base64, DecodeHelloWorld) {
    auto result = Base64::decode("SGVsbG8sIFdvcmxkIQ==");
    std::string str(result.begin(), result.end());
    EXPECT_EQ(str, "Hello, World!");
}

TEST(Base64, DecodeBinaryData) {
    auto result = Base64::decode("AP9/gAE=");
    ASSERT_EQ(result.size(), 5u);
    EXPECT_EQ(result[0], 0x00);
    EXPECT_EQ(result[1], 0xFF);
    EXPECT_EQ(result[2], 0x7F);
    EXPECT_EQ(result[3], 0x80);
    EXPECT_EQ(result[4], 0x01);
}

// ============================================================
// Round-trip Tests
// ============================================================

TEST(Base64, RoundTripSmall) {
    std::vector<uint8_t> original = {0x01, 0x02, 0x03, 0x04, 0x05};
    auto encoded = Base64::encode(original);
    auto decoded = Base64::decode(encoded);
    EXPECT_EQ(original, decoded);
}

TEST(Base64, RoundTripLarge) {
    // Simulate filter data (~1KB)
    std::vector<uint8_t> original(1024);
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = static_cast<uint8_t>(i & 0xFF);
    }
    auto encoded = Base64::encode(original);
    auto decoded = Base64::decode(encoded);
    EXPECT_EQ(original, decoded);
}

TEST(Base64, RoundTripAllByteValues) {
    // Test all 256 byte values
    std::vector<uint8_t> original(256);
    for (int i = 0; i < 256; ++i) {
        original[i] = static_cast<uint8_t>(i);
    }
    auto encoded = Base64::encode(original);
    auto decoded = Base64::decode(encoded);
    EXPECT_EQ(original, decoded);
}

// ============================================================
// Invalid Input Tests
// ============================================================

TEST(Base64, DecodeInvalidLength) {
    // Not multiple of 4
    EXPECT_TRUE(Base64::decode("ABC").empty());
    EXPECT_TRUE(Base64::decode("ABCDE").empty());
}

TEST(Base64, DecodeInvalidCharacter) {
    // Contains invalid character '!'
    EXPECT_TRUE(Base64::decode("ABC!").empty());
    // Contains invalid character ' '
    EXPECT_TRUE(Base64::decode("AB C").empty());
}

TEST(Base64, DecodeInvalidPadding) {
    // Padding in wrong position
    EXPECT_TRUE(Base64::decode("=ABC").empty());
    EXPECT_TRUE(Base64::decode("A=BC").empty());
}

// ============================================================
// Validation Tests
// ============================================================

TEST(Base64, IsValidEmpty) {
    EXPECT_TRUE(Base64::isValid(""));
}

TEST(Base64, IsValidCorrect) {
    EXPECT_TRUE(Base64::isValid("TWFu"));
    EXPECT_TRUE(Base64::isValid("TWE="));
    EXPECT_TRUE(Base64::isValid("TQ=="));
    EXPECT_TRUE(Base64::isValid("SGVsbG8sIFdvcmxkIQ=="));
}

TEST(Base64, IsValidInvalidLength) {
    EXPECT_FALSE(Base64::isValid("ABC"));
    EXPECT_FALSE(Base64::isValid("ABCDE"));
}

TEST(Base64, IsValidInvalidCharacter) {
    EXPECT_FALSE(Base64::isValid("ABC!"));
    EXPECT_FALSE(Base64::isValid("AB C"));
}

// ============================================================
// Size Calculation Tests
// ============================================================

TEST(Base64, EncodedSize) {
    EXPECT_EQ(Base64::encodedSize(0), 0u);
    EXPECT_EQ(Base64::encodedSize(1), 4u);
    EXPECT_EQ(Base64::encodedSize(2), 4u);
    EXPECT_EQ(Base64::encodedSize(3), 4u);
    EXPECT_EQ(Base64::encodedSize(4), 8u);
    EXPECT_EQ(Base64::encodedSize(5), 8u);
    EXPECT_EQ(Base64::encodedSize(6), 8u);
}

TEST(Base64, DecodedSize) {
    EXPECT_EQ(Base64::decodedSize(""), 0u);
    EXPECT_EQ(Base64::decodedSize("TQ=="), 1u);
    EXPECT_EQ(Base64::decodedSize("TWE="), 2u);
    EXPECT_EQ(Base64::decodedSize("TWFu"), 3u);
    // Invalid length returns 0
    EXPECT_EQ(Base64::decodedSize("ABC"), 0u);
}
