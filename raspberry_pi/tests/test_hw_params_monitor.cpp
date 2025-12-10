// Tests for HwParamsMonitor parsing via override path
#include "HwParamsMonitor.h"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

namespace fs = std::filesystem;

namespace {

fs::path writeTemp(const std::string &content) {
    auto tmp = fs::temp_directory_path() / fs::path("hw_params_test.txt");
    std::ofstream out(tmp);
    out << content;
    return tmp;
}

}  // namespace

TEST(HwParamsMonitor, ParsesValidHwParams) {
    const auto path = writeTemp(
        "access: RW_INTERLEAVED\n"
        "format: S24_3LE\n"
        "subformat: STD\n"
        "channels: 2\n"
        "rate: 96000\n");
    HwParamsMonitor monitor("hw:0,0", path.string());
    auto params = monitor.readCurrent();

    ASSERT_TRUE(params.has_value());
    EXPECT_EQ(params->sampleRate, 96000u);
    EXPECT_EQ(params->channels, 2u);
    EXPECT_EQ(params->format, AlsaCapture::SampleFormat::S24_3LE);
}

TEST(HwParamsMonitor, RejectsUnsupportedRate) {
    const auto path = writeTemp("format: S16_LE\nchannels: 2\nrate: 12345\n");
    HwParamsMonitor monitor("hw:0,0", path.string());
    auto params = monitor.readCurrent();

    EXPECT_FALSE(params.has_value());
}

TEST(HwParamsMonitor, RejectsUnknownFormat) {
    const auto path = writeTemp("format: S20_LE\nchannels: 2\nrate: 48000\n");
    HwParamsMonitor monitor("hw:0,0", path.string());
    auto params = monitor.readCurrent();

    EXPECT_FALSE(params.has_value());
}
