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

    ASSERT_FALSE(parsed.hasError);
    ASSERT_TRUE(parsed.options.has_value());
    EXPECT_EQ(parsed.options->device, "hw:0,0");
    EXPECT_EQ(parsed.options->host, "127.0.0.1");
    EXPECT_EQ(parsed.options->port, 46001);
    EXPECT_EQ(parsed.options->rate, 48000u);
    EXPECT_EQ(parsed.options->format, AlsaCapture::SampleFormat::S16_LE);
    EXPECT_EQ(parsed.options->frames, static_cast<snd_pcm_uframes_t>(4096));
    EXPECT_EQ(parsed.options->logLevel, LogLevel::Info);
    EXPECT_EQ(parsed.options->iterations, -1);
}

TEST(ParseOptions, ParsesProvidedArguments) {
    std::vector<std::string> args = {
        "rpi_pcm_bridge", "-d",           "hw:2,0", "-H",          "192.168.0.10", "-p",
        "55000",          "-r",           "96000",  "-f",          "S24_3LE",      "--frames",
        "2048",           "--iterations", "5",      "--log-level", "debug"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    ASSERT_FALSE(parsed.hasError);
    ASSERT_TRUE(parsed.options.has_value());
    EXPECT_EQ(parsed.options->device, "hw:2,0");
    EXPECT_EQ(parsed.options->host, "192.168.0.10");
    EXPECT_EQ(parsed.options->port, 55000);
    EXPECT_EQ(parsed.options->rate, 96000u);
    EXPECT_EQ(parsed.options->format, AlsaCapture::SampleFormat::S24_3LE);
    EXPECT_EQ(parsed.options->frames, static_cast<snd_pcm_uframes_t>(2048));
    EXPECT_EQ(parsed.options->logLevel, LogLevel::Debug);
    EXPECT_EQ(parsed.options->iterations, 5);
}

TEST(ParseOptions, RejectsInvalidFormat) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--format", "S20_LE"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    EXPECT_TRUE(parsed.hasError);
    EXPECT_FALSE(parsed.options.has_value());
}

TEST(ParseOptions, RejectsInvalidRate) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--rate", "12345"};
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

TEST(ParseOptions, RejectsInvalidLogLevel) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--log-level", "verbose"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    EXPECT_TRUE(parsed.hasError);
    EXPECT_FALSE(parsed.options.has_value());
}

TEST(ParseOptions, ShowsHelp) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--help"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    EXPECT_TRUE(parsed.showHelp);
    EXPECT_FALSE(parsed.hasError);
}

TEST(ParseOptions, ShowsVersion) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--version"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    EXPECT_TRUE(parsed.showVersion);
    EXPECT_FALSE(parsed.hasError);
}

TEST(ParseOptions, RejectsUnknownFlag) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--unknown", "value"};
    auto argv = makeArgv(args);
    auto parsed = parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge");

    EXPECT_TRUE(parsed.hasError);
    EXPECT_FALSE(parsed.options.has_value());
}
