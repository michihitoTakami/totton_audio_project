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

    EXPECT_FALSE(parsed.showHelp);
    EXPECT_FALSE(parsed.showVersion);
    EXPECT_FALSE(parsed.hasError);
    ASSERT_TRUE(parsed.options.has_value());
    EXPECT_EQ(parsed.options->device, "hw:0,0");
    EXPECT_EQ(parsed.options->host, "127.0.0.1");
    EXPECT_EQ(parsed.options->port, 46001);
    EXPECT_EQ(parsed.options->rate, 48000u);
    EXPECT_EQ(parsed.options->format, AlsaCapture::SampleFormat::S16_LE);
    EXPECT_EQ(parsed.options->frames, static_cast<snd_pcm_uframes_t>(4096));
    EXPECT_EQ(parsed.options->iterations, 3);
    EXPECT_EQ(parsed.options->logLevel, Options::LogLevel::Info);
}

TEST(ParseOptions, ParsesProvidedArguments) {
    std::vector<std::string> args = {
        "rpi_pcm_bridge", "-d",           "hw:2,0", "-H", "192.168.55.1", "-p",
        "55000",          "-r",           "96000",  "-f", "S24_3LE",      "--frames",
        "2048",           "--iterations", "5",      "-l", "warn"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    ASSERT_TRUE(parsed.options.has_value());
    const auto &opt = *parsed.options;
    EXPECT_EQ(opt.device, "hw:2,0");
    EXPECT_EQ(opt.host, "192.168.55.1");
    EXPECT_EQ(opt.port, 55000);
    EXPECT_EQ(opt.rate, 96000u);
    EXPECT_EQ(opt.format, AlsaCapture::SampleFormat::S24_3LE);
    EXPECT_EQ(opt.frames, static_cast<snd_pcm_uframes_t>(2048));
    EXPECT_EQ(opt.iterations, 5);
    EXPECT_EQ(opt.logLevel, Options::LogLevel::Warn);
}

TEST(ParseOptions, AcceptsSupportedRates) {
    for (const auto rate :
         {44100u, 48000u, 88200u, 96000u, 176400u, 192000u, 352800u, 384000u, 705600u, 768000u}) {
        std::vector<std::string> args = {"rpi_pcm_bridge", "--rate", std::to_string(rate)};
        auto argv = makeArgv(args);
        auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");
        ASSERT_TRUE(parsed.options.has_value());
        EXPECT_EQ(parsed.options->rate, rate);
    }
}

TEST(ParseOptions, RejectsInvalidFormat) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--format", "S20_LE"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    EXPECT_TRUE(parsed.hasError);
    EXPECT_FALSE(parsed.options.has_value());
}

TEST(ParseOptions, RejectsUnknownFlag) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--unknown", "value"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    EXPECT_TRUE(parsed.hasError);
    EXPECT_FALSE(parsed.options.has_value());
}

TEST(ParseOptions, RejectsUnsupportedRate) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--rate", "50000"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    EXPECT_TRUE(parsed.hasError);
    EXPECT_FALSE(parsed.options.has_value());
}

TEST(ParseOptions, RejectsInvalidPort) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--port", "70000"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    EXPECT_TRUE(parsed.hasError);
    EXPECT_FALSE(parsed.options.has_value());
}

TEST(ParseOptions, AcceptsVersionFlag) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--version"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    EXPECT_TRUE(parsed.showVersion);
    EXPECT_FALSE(parsed.options.has_value());
}
