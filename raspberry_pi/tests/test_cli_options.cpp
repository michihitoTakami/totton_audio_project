#include "Options.h"

#include <array>
#include <functional>
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
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

using EnvMap = std::unordered_map<std::string, std::string>;

std::function<const char *(const char *)> makeGetEnv(const EnvMap &env) {
    return [&env](const char *name) -> const char * {
        auto it = env.find(name ? std::string{name} : std::string{});
        if (it == env.end()) {
            return nullptr;
        }
        return it->second.c_str();
    };
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

TEST(ParseOptions, AppliesEnvironmentOverrides) {
    std::vector<std::string> args = {"rpi_pcm_bridge"};
    auto argv = makeArgv(args);

    EnvMap env = {{"PCM_BRIDGE_DEVICE", "hw:9,0"},   {"PCM_BRIDGE_HOST", "10.0.0.2"},
                  {"PCM_BRIDGE_PORT", "47001"},      {"PCM_BRIDGE_RATE", "192000"},
                  {"PCM_BRIDGE_FORMAT", "S32_LE"},   {"PCM_BRIDGE_FRAMES", "1024"},
                  {"PCM_BRIDGE_LOG_LEVEL", "debug"}, {"PCM_BRIDGE_ITERATIONS", "5"}};

    auto parsed =
        parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge", makeGetEnv(env));

    ASSERT_FALSE(parsed.hasError);
    ASSERT_TRUE(parsed.options.has_value());
    EXPECT_EQ(parsed.options->device, "hw:9,0");
    EXPECT_EQ(parsed.options->host, "10.0.0.2");
    EXPECT_EQ(parsed.options->port, 47001);
    EXPECT_EQ(parsed.options->rate, 192000u);
    EXPECT_EQ(parsed.options->format, AlsaCapture::SampleFormat::S32_LE);
    EXPECT_EQ(parsed.options->frames, static_cast<snd_pcm_uframes_t>(1024));
    EXPECT_EQ(parsed.options->logLevel, LogLevel::Debug);
    EXPECT_EQ(parsed.options->iterations, 5);
}

TEST(ParseOptions, RejectsInvalidEnvironmentRate) {
    std::vector<std::string> args = {"rpi_pcm_bridge"};
    auto argv = makeArgv(args);

    EnvMap env = {{"PCM_BRIDGE_RATE", "12345"}};
    auto parsed =
        parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge", makeGetEnv(env));

    EXPECT_TRUE(parsed.hasError);
    EXPECT_FALSE(parsed.options.has_value());
}

TEST(ParseOptions, CliOverridesEnvironment) {
    std::vector<std::string> args = {"rpi_pcm_bridge", "--log-level", "info", "--port", "55000"};
    auto argv = makeArgv(args);

    EnvMap env = {{"PCM_BRIDGE_LOG_LEVEL", "debug"}, {"PCM_BRIDGE_PORT", "47001"}};
    auto parsed =
        parseOptions(static_cast<int>(argv.size()), argv.data(), "rpi_pcm_bridge", makeGetEnv(env));

    ASSERT_FALSE(parsed.hasError);
    ASSERT_TRUE(parsed.options.has_value());
    EXPECT_EQ(parsed.options->logLevel, LogLevel::Info);  // CLI wins
    EXPECT_EQ(parsed.options->port, 55000);               // CLI wins
}
