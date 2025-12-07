#include "Options.h"

#include <array>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

std::vector<char *> makeArgv(const std::vector<std::string> &args) {
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (const auto &arg : args) {
        argv.push_back(const_cast<char *>(arg.c_str()));
    }
    return argv;
}

}  // namespace

TEST(ParseFormat, AcceptsSupportedValues) {
    EXPECT_EQ(parseFormat("S16_LE"), AlsaCapture::SampleFormat::S16_LE);
    EXPECT_EQ(parseFormat("S24_3LE"), AlsaCapture::SampleFormat::S24_3LE);
    EXPECT_EQ(parseFormat("S32_LE"), AlsaCapture::SampleFormat::S32_LE);
}

TEST(ParseFormat, RejectsUnknownValues) {
    EXPECT_FALSE(parseFormat("PCM24"));
    EXPECT_FALSE(parseFormat("S24_LE"));
}

TEST(ParseOptions, ReturnsDefaultsWhenNoArgs) {
    std::vector<std::string> args = {"rpi_pcm_bridge"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->device, "hw:0,0");
    EXPECT_EQ(parsed->rate, 48000u);
    EXPECT_EQ(parsed->format, AlsaCapture::SampleFormat::S16_LE);
    EXPECT_EQ(parsed->frames, static_cast<snd_pcm_uframes_t>(4096));
    EXPECT_EQ(parsed->iterations, 3);
}

TEST(ParseOptions, ParsesProvidedArguments) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--device",     "hw:2,0",  "--rate",
                                     "96000",          "--format",     "S24_3LE", "--frames",
                                     "2048",           "--iterations", "5"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->device, "hw:2,0");
    EXPECT_EQ(parsed->rate, 96000u);
    EXPECT_EQ(parsed->format, AlsaCapture::SampleFormat::S24_3LE);
    EXPECT_EQ(parsed->frames, static_cast<snd_pcm_uframes_t>(2048));
    EXPECT_EQ(parsed->iterations, 5);
}

TEST(ParseOptions, RejectsInvalidFormat) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--format", "S20_LE"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    EXPECT_FALSE(parsed.has_value());
}

TEST(ParseOptions, RejectsUnknownFlag) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--unknown", "value"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    EXPECT_FALSE(parsed.has_value());
}
