#include "AlsaCapture.h"

#include <gtest/gtest.h>

TEST(AlsaCaptureHelpers, MapsFormatsToAlsaEnums) {
    EXPECT_EQ(AlsaCapture::toAlsaFormat(AlsaCapture::SampleFormat::S16_LE), SND_PCM_FORMAT_S16_LE);
    EXPECT_EQ(AlsaCapture::toAlsaFormat(AlsaCapture::SampleFormat::S24_3LE),
              SND_PCM_FORMAT_S24_3LE);
    EXPECT_EQ(AlsaCapture::toAlsaFormat(AlsaCapture::SampleFormat::S32_LE), SND_PCM_FORMAT_S32_LE);
}

TEST(AlsaCaptureHelpers, CalculatesBytesPerFrame) {
    AlsaCapture::Config config{};
    config.channels = 2;

    config.format = AlsaCapture::SampleFormat::S16_LE;
    EXPECT_EQ(AlsaCapture::bytesPerFrame(config), 4U);

    config.format = AlsaCapture::SampleFormat::S24_3LE;
    EXPECT_EQ(AlsaCapture::bytesPerFrame(config), 6U);

    config.format = AlsaCapture::SampleFormat::S32_LE;
    EXPECT_EQ(AlsaCapture::bytesPerFrame(config), 8U);
}
