#include "delimiter/inference_backend.h"
#include "gtest/gtest.h"

#include <vector>

TEST(DelimiterInferenceBackend, BypassCopiesInput) {
    AppConfig::DelimiterConfig cfg;
    cfg.enabled = true;
    cfg.backend = "bypass";

    auto backend = delimiter::createDelimiterInferenceBackend(cfg);
    ASSERT_NE(backend, nullptr);
    EXPECT_STREQ(backend->name(), "bypass");
    EXPECT_EQ(backend->expectedSampleRate(), 44100u);

    std::vector<float> leftIn = {0.1f, -0.2f, 0.3f};
    std::vector<float> rightIn = {-0.4f, 0.5f, -0.6f};

    delimiter::StereoPlanarView input;
    input.left = leftIn.data();
    input.right = rightIn.data();
    input.frames = leftIn.size();

    std::vector<float> leftOut;
    std::vector<float> rightOut;

    auto result = backend->process(input, leftOut, rightOut);
    EXPECT_EQ(result.status, delimiter::InferenceStatus::Ok);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(leftOut, leftIn);
    EXPECT_EQ(rightOut, rightIn);
}

TEST(DelimiterInferenceBackend, DisabledAlwaysBypasses) {
    AppConfig::DelimiterConfig cfg;
    cfg.enabled = false;
    cfg.backend = "ort";

    auto backend = delimiter::createDelimiterInferenceBackend(cfg);
    ASSERT_NE(backend, nullptr);
    EXPECT_STREQ(backend->name(), "bypass");
}
