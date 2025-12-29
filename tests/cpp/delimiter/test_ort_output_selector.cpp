#include "delimiter/ort_output_selector.h"
#include "gtest/gtest.h"

#include <vector>

TEST(OrtOutputSelector, PrefersDecodedWhenStereo) {
    std::vector<std::string> names = {"enhanced", "decoded"};
    std::vector<std::vector<int64_t>> shapes = {{1, 1, 256}, {1, 2, 256}};
    auto idx = delimiter::pickStereoOutputIndex(names, shapes);
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(*idx, 1u);
}

TEST(OrtOutputSelector, PicksFirstStereoWhenNoDecoded) {
    std::vector<std::string> names = {"foo", "bar"};
    std::vector<std::vector<int64_t>> shapes = {{1, 2, 256}, {1, 2, 256}};
    auto idx = delimiter::pickStereoOutputIndex(names, shapes);
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(*idx, 0u);
}

TEST(OrtOutputSelector, IgnoresDecodedIfNotStereo) {
    std::vector<std::string> names = {"decoded", "other"};
    std::vector<std::vector<int64_t>> shapes = {{1, 1, 256}, {2, 256}};
    auto idx = delimiter::pickStereoOutputIndex(names, shapes);
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(*idx, 1u);
}

TEST(OrtOutputSelector, ReturnsNulloptIfNoStereoLikeOutput) {
    std::vector<std::string> names = {"enhanced", "mask"};
    std::vector<std::vector<int64_t>> shapes = {{1, 1, 256}, {256}};
    auto idx = delimiter::pickStereoOutputIndex(names, shapes);
    EXPECT_FALSE(idx.has_value());
}
