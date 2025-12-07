#include "pcm_header.h"

#include <cstring>
#include <gtest/gtest.h>

namespace {

PcmHeader makeValidHeader() {
    PcmHeader h{};
    std::memcpy(h.magic, "PCMA", 4);
    h.version = 1;
    h.sample_rate = 48000;
    h.channels = 2;
    h.format = 1;
    return h;
}

}  // namespace

TEST(PcmHeaderValidation, AcceptsValidHeader) {
    auto header = makeValidHeader();
    auto result = validateHeader(header);

    EXPECT_TRUE(result.ok);
    EXPECT_TRUE(result.reason.empty());
}

TEST(PcmHeaderValidation, RejectsInvalidMagic) {
    auto header = makeValidHeader();
    std::memcpy(header.magic, "BAD!", 4);

    auto result = validateHeader(header);

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.reason, "magic != PCMA");
}

TEST(PcmHeaderValidation, RejectsWrongVersion) {
    auto header = makeValidHeader();
    header.version = 2;

    auto result = validateHeader(header);

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.reason, "version != 1");
}

TEST(PcmHeaderValidation, RejectsOutOfRangeRate) {
    auto header = makeValidHeader();
    header.sample_rate = 32000;
    auto lowResult = validateHeader(header);

    EXPECT_FALSE(lowResult.ok);
    EXPECT_EQ(lowResult.reason, "sample_rate unsupported");

    header.sample_rate = 800000;
    auto highResult = validateHeader(header);

    EXPECT_FALSE(highResult.ok);
    EXPECT_EQ(highResult.reason, "sample_rate unsupported");
}

TEST(PcmHeaderValidation, RejectsOutOfRangeChannels) {
    auto header = makeValidHeader();
    header.channels = 0;
    auto lowResult = validateHeader(header);

    EXPECT_FALSE(lowResult.ok);
    EXPECT_EQ(lowResult.reason, "channels unsupported");

    header.channels = 16;
    auto highResult = validateHeader(header);

    EXPECT_FALSE(highResult.ok);
    EXPECT_EQ(highResult.reason, "channels unsupported");
}

TEST(PcmHeaderValidation, RejectsUnsupportedFormat) {
    auto header = makeValidHeader();
    header.format = 99;

    auto result = validateHeader(header);

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.reason, "unsupported format");
}
