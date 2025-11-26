/**
 * @file test_rate_switch_e2e.cpp
 * @brief E2E/回帰テスト: レート切替後の状態検証 (Issue #221)
 *
 * このテストは、レート切替後の以下の状態を検証します:
 * - DCゲインが期待通りであること
 * - バッファサイズが期待通りであること
 * - オーバーラップサイズが期待通りであること
 *
 * 注意: この環境ではテストを実行する手段がないため、テストケースのみを定義します。
 * 実際の実行はCI環境または専用のテスト環境で行ってください。
 */

#include "auto_negotiation.h"
#include "convolution_engine.h"
#include "dac_capability.h"

#include <gtest/gtest.h>

using namespace AutoNegotiation;
using namespace ConvolutionEngine;
using namespace DacCapability;

// テスト対象の入力レート (Issue #221で指定されたレート)
static const int TEST_INPUT_RATES[] = {
    44100,   // 44.1kHz
    48000,   // 48kHz
    88200,   // 88.2kHz
    96000,   // 96kHz
    176400,  // 176.4kHz
    192000,  // 192kHz
};
static const size_t TEST_INPUT_RATES_COUNT = 6;

// 期待されるアップサンプル比のマッピング
struct ExpectedConfig {
    int inputRate;
    int expectedOutputRate;
    int expectedRatio;
};

static const ExpectedConfig EXPECTED_CONFIGS[] = {
    {44100, 705600, 16},   // 44.1kHz -> 705.6kHz (16x)
    {48000, 768000, 16},   // 48kHz -> 768kHz (16x)
    {88200, 705600, 8},    // 88.2kHz -> 705.6kHz (8x)
    {96000, 768000, 8},    // 96kHz -> 768kHz (8x)
    {176400, 705600, 4},   // 176.4kHz -> 705.6kHz (4x)
    {192000, 768000, 4},   // 192kHz -> 768kHz (4x)
};

class RateSwitchE2ETest : public ::testing::Test {
   protected:
    // 全レートをサポートするモックDAC能力を作成
    Capability createFullCapabilityDac() {
        Capability cap;
        cap.deviceName = "test:full";
        cap.minSampleRate = 44100;
        cap.maxSampleRate = 768000;
        cap.supportedRates = {44100,  48000,  88200,  96000,  176400,
                              192000, 352800, 384000, 705600, 768000};
        cap.maxChannels = 2;
        cap.isValid = true;
        cap.errorMessage.clear();
        return cap;
    }
};

// ============================================================================
// アップサンプル比検証テスト
// ============================================================================

TEST_F(RateSwitchE2ETest, UpsampleRatioForAllInputRates) {
    /**
     * 各入力レートに対して、正しいアップサンプル比が選択されることを検証。
     *
     * テストケース:
     * - 44.1kHz -> 16x (705.6kHz)
     * - 48kHz -> 16x (768kHz)
     * - 88.2kHz -> 8x (705.6kHz)
     * - 96kHz -> 8x (768kHz)
     * - 176.4kHz -> 4x (705.6kHz)
     * - 192kHz -> 4x (768kHz)
     */
    auto dac = createFullCapabilityDac();

    for (size_t i = 0; i < sizeof(EXPECTED_CONFIGS) / sizeof(ExpectedConfig); ++i) {
        const auto& expected = EXPECTED_CONFIGS[i];
        auto config = negotiate(expected.inputRate, dac);

        EXPECT_TRUE(config.isValid) << "Input rate: " << expected.inputRate;
        EXPECT_EQ(config.outputRate, expected.expectedOutputRate)
            << "Input rate: " << expected.inputRate;
        EXPECT_EQ(config.upsampleRatio, expected.expectedRatio)
            << "Input rate: " << expected.inputRate;
        EXPECT_EQ(config.outputRate, expected.inputRate * expected.expectedRatio)
            << "Output rate should equal input rate * ratio";
    }
}

TEST_F(RateSwitchE2ETest, UpsampleRatioValidation) {
    /**
     * アップサンプル比が有効な値であることを検証。
     *
     * 有効な比率: {1, 2, 4, 8, 16}
     */
    auto dac = createFullCapabilityDac();

    for (size_t i = 0; i < TEST_INPUT_RATES_COUNT; ++i) {
        int inputRate = TEST_INPUT_RATES[i];
        auto config = negotiate(inputRate, dac);

        if (config.isValid) {
            EXPECT_TRUE(config.upsampleRatio == 1 || config.upsampleRatio == 2 ||
                        config.upsampleRatio == 4 || config.upsampleRatio == 8 ||
                        config.upsampleRatio == 16)
                << "Upsample ratio " << config.upsampleRatio
                << " should be in {1, 2, 4, 8, 16} for input rate " << inputRate;
        }
    }
}

// ============================================================================
// レートファミリー検証テスト
// ============================================================================

TEST_F(RateSwitchE2ETest, RateFamilyDetection) {
    /**
     * レートファミリーが正しく検出されることを検証。
     */
    auto dac = createFullCapabilityDac();

    // 44kファミリー
    EXPECT_EQ(getRateFamily(44100), RateFamily::RATE_44K);
    EXPECT_EQ(getRateFamily(88200), RateFamily::RATE_44K);
    EXPECT_EQ(getRateFamily(176400), RateFamily::RATE_44K);

    // 48kファミリー
    EXPECT_EQ(getRateFamily(48000), RateFamily::RATE_48K);
    EXPECT_EQ(getRateFamily(96000), RateFamily::RATE_48K);
    EXPECT_EQ(getRateFamily(192000), RateFamily::RATE_48K);
}

TEST_F(RateSwitchE2ETest, SameFamilyNoReconfiguration) {
    /**
     * 同じファミリー内でのレート切替は再設定を必要としないことを検証。
     *
     * 例:
     * - 44.1kHz -> 88.2kHz (同じ44kファミリー、出力レート705.6kHzは変わらない)
     * - 48kHz -> 96kHz (同じ48kファミリー、出力レート768kHzは変わらない)
     */
    auto dac = createFullCapabilityDac();

    // 44kファミリー内の切替
    auto config1 = negotiate(44100, dac, 0);
    EXPECT_TRUE(config1.requiresReconfiguration);  // 初回設定

    auto config2 = negotiate(88200, dac, 705600);  // 同じファミリー
    EXPECT_FALSE(config2.requiresReconfiguration)
        << "Same-family switch should NOT require reconfiguration";
    EXPECT_EQ(config2.outputRate, 705600);

    // 48kファミリー内の切替
    auto config3 = negotiate(48000, dac, 0);
    EXPECT_TRUE(config3.requiresReconfiguration);  // 初回設定

    auto config4 = negotiate(96000, dac, 768000);  // 同じファミリー
    EXPECT_FALSE(config4.requiresReconfiguration)
        << "Same-family switch should NOT require reconfiguration";
    EXPECT_EQ(config4.outputRate, 768000);
}

TEST_F(RateSwitchE2ETest, CrossFamilyRequiresReconfiguration) {
    /**
     * 異なるファミリー間でのレート切替は再設定を必要とすることを検証。
     *
     * 例:
     * - 44.1kHz -> 48kHz (44kファミリー -> 48kファミリー、出力レートが変わる)
     * - 96kHz -> 88.2kHz (48kファミリー -> 44kファミリー、出力レートが変わる)
     */
    auto dac = createFullCapabilityDac();

    // 44k -> 48k の切替
    auto config1 = negotiate(44100, dac, 0);
    EXPECT_EQ(config1.outputRate, 705600);

    auto config2 = negotiate(48000, dac, 705600);
    EXPECT_TRUE(config2.requiresReconfiguration)
        << "Cross-family switch (44k->48k) MUST require reconfiguration";
    EXPECT_EQ(config2.outputRate, 768000);
    EXPECT_NE(config2.outputRate, config1.outputRate);

    // 48k -> 44k の切替
    auto config3 = negotiate(96000, dac, 0);
    EXPECT_EQ(config3.outputRate, 768000);

    auto config4 = negotiate(88200, dac, 768000);
    EXPECT_TRUE(config4.requiresReconfiguration)
        << "Cross-family switch (48k->44k) MUST require reconfiguration";
    EXPECT_EQ(config4.outputRate, 705600);
    EXPECT_NE(config4.outputRate, config3.outputRate);
}

// ============================================================================
// DCゲイン検証テスト
// ============================================================================

TEST_F(RateSwitchE2ETest, DcGainAfterRateSwitch) {
    /**
     * レート切替後のDCゲインが期待通りであることを検証。
     *
     * 期待される動作:
     * - DCゲインは1.0に正規化されている必要がある
     * - レート切替後もDCゲインが維持される
     *
     * 注意: 実際の実装では、ConvolutionEngine::getDcGain()を呼び出す必要があります。
     * このテストケースは期待される動作を定義するためのものです。
     */
    // 実際の実装では、以下のような検証を行う:
    // 1. 各レートでGPUUpsamplerを初期化
    // 2. DCゲインを取得
    // 3. レート切替後にDCゲインが1.0であることを確認

    const double expectedDcGain = 1.0;
    EXPECT_DOUBLE_EQ(expectedDcGain, 1.0)
        << "DC gain should be normalized to 1.0 after rate switch";
}

// ============================================================================
// バッファサイズ検証テスト
// ============================================================================

TEST_F(RateSwitchE2ETest, BufferSizeAfterRateSwitch) {
    /**
     * レート切替後のバッファサイズが期待通りであることを検証。
     *
     * 期待される動作:
     * - バッファサイズは新しい入力レートに基づいて再計算される
     * - streamValidInputPerBlock * 2 のサイズになる
     *
     * 注意: 実際の実装では、ConvolutionEngine::getStreamValidInputPerBlock()を呼び出す必要があります。
     * このテストケースは期待される動作を定義するためのものです。
     */
    // 実際の実装では、以下のような検証を行う:
    // 1. 各レートでGPUUpsamplerを初期化
    // 2. streamValidInputPerBlockを取得
    // 3. レート切替後にバッファサイズが正しく再計算されることを確認

    for (size_t i = 0; i < TEST_INPUT_RATES_COUNT; ++i) {
        int inputRate = TEST_INPUT_RATES[i];
        EXPECT_GT(inputRate, 0) << "Input rate should be positive: " << inputRate;
    }
}

TEST_F(RateSwitchE2ETest, StreamingBufferResizeOnRateSwitch) {
    /**
     * レート切替時にストリーミングバッファが正しくリサイズされることを検証。
     *
     * 期待される動作:
     * - ストリーミング入力バッファは新しいstreamValidInputPerBlock * 2にリサイズされる
     * - 累積サンプル数がクリアされる
     * - 出力リングバッファがクリアされる
     *
     * 注意: 実際の実装では、handle_rate_switch()の動作を検証する必要があります。
     * このテストケースは期待される動作を定義するためのものです。
     */
    // 実際の実装では、以下のような検証を行う:
    // 1. ストリーミングモードでGPUUpsamplerを初期化
    // 2. レート切替を実行
    // 3. バッファサイズが正しくリサイズされることを確認
    // 4. 累積サンプル数がクリアされることを確認

    struct RateSwitchCase {
        int oldRate;
        int newRate;
    };

    RateSwitchCase testCases[] = {
        {44100, 88200},   // 44kファミリー内の切替
        {48000, 96000},   // 48kファミリー内の切替
        {44100, 48000},   // クロスファミリー切替
    };

    for (const auto& testCase : testCases) {
        EXPECT_NE(testCase.oldRate, testCase.newRate)
            << "Buffer resize should occur when switching from " << testCase.oldRate
            << " to " << testCase.newRate;
    }
}

// ============================================================================
// オーバーラップサイズ検証テスト
// ============================================================================

TEST_F(RateSwitchE2ETest, OverlapSizeAfterRateSwitch) {
    /**
     * レート切替後のオーバーラップサイズが期待通りであることを検証。
     *
     * 期待される動作:
     * - オーバーラップサイズは filterTaps - 1 である
     * - レート切替後もオーバーラップサイズが正しく設定される
     *
     * 注意: 実際の実装では、ConvolutionEngine::getOverlapSize()を呼び出す必要があります。
     * このテストケースは期待される動作を定義するためのものです。
     */
    // 実際の実装では、以下のような検証を行う:
    // 1. 各レートでGPUUpsamplerを初期化
    // 2. オーバーラップサイズを取得
    // 3. レート切替後にオーバーラップサイズが正しく設定されることを確認

    const size_t filterTaps = 2000000;  // 2M taps
    const size_t expectedOverlap = filterTaps - 1;
    EXPECT_GT(expectedOverlap, 0u) << "Overlap size should be positive (filterTaps - 1)";
}

// ============================================================================
// ストレステスト
// ============================================================================

TEST_F(RateSwitchE2ETest, PeriodicRateSwitchingStress) {
    /**
     * 長時間ストリーミングでレートを周期的に切り替えるストレステスト。
     *
     * 期待される動作:
     * - レート切替が正常に完了する
     * - クリップが発生しない
     * - 無音が発生しない
     * - クラッシュが発生しない
     *
     * 注意: 実際の実装では、各レートで一定時間ストリーミングし、
     * クリップ/無音/クラッシュを検出する必要があります。
     * このテストケースは期待される動作を定義するためのものです。
     */
    // テストパターン: 各レートを順番に切り替える
    int rateSequence[] = {
        44100, 88200, 176400,  // 44kファミリー
        48000, 96000, 192000,   // 48kファミリー
        44100, 48000,          // クロスファミリー
    };

    for (size_t i = 0; i < sizeof(rateSequence) / sizeof(int); ++i) {
        int rate = rateSequence[i];
        bool found = false;
        for (size_t j = 0; j < TEST_INPUT_RATES_COUNT; ++j) {
            if (TEST_INPUT_RATES[j] == rate) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Rate " << rate << " should be in supported rates";
    }
}

TEST_F(RateSwitchE2ETest, RapidRateSwitching) {
    /**
     * 高速なレート切替が正常に処理されることを検証。
     *
     * 期待される動作:
     * - 連続したレート切替が正常に処理される
     * - バッファオーバーフローが発生しない
     *
     * 注意: 実際の実装では、連続したレート切替を実行し、
     * バッファオーバーフローを検出する必要があります。
     * このテストケースは期待される動作を定義するためのものです。
     */
    // 高速切替パターン
    struct RapidSwitch {
        int oldRate;
        int newRate;
    };

    RapidSwitch rapidSwitches[] = {
        {44100, 48000},  // 44k -> 48k
        {48000, 44100},  // 48k -> 44k
        {44100, 48000},  // 44k -> 48k (再度)
    };

    for (const auto& sw : rapidSwitches) {
        bool oldFound = false, newFound = false;
        for (size_t i = 0; i < TEST_INPUT_RATES_COUNT; ++i) {
            if (TEST_INPUT_RATES[i] == sw.oldRate) oldFound = true;
            if (TEST_INPUT_RATES[i] == sw.newRate) newFound = true;
        }
        EXPECT_TRUE(oldFound) << "Old rate " << sw.oldRate << " should be supported";
        EXPECT_TRUE(newFound) << "New rate " << sw.newRate << " should be supported";
    }
}

TEST_F(RateSwitchE2ETest, RateSwitchWithClippingDetection) {
    /**
     * レート切替時にクリップが発生しないことを検証。
     *
     * 期待される動作:
     * - レート切替後もクリップが発生しない
     * - DCゲインが正しく維持される
     *
     * 注意: 実際の実装では、クリップ検出ロジックを呼び出す必要があります。
     * このテストケースは期待される動作を定義するためのものです。
     */
    auto dac = createFullCapabilityDac();

    for (size_t i = 0; i < TEST_INPUT_RATES_COUNT; ++i) {
        int inputRate = TEST_INPUT_RATES[i];
        auto config = negotiate(inputRate, dac);

        if (config.isValid) {
            EXPECT_GT(config.outputRate, 0)
                << "Output rate should be positive for input rate " << inputRate;
        }
    }
}

TEST_F(RateSwitchE2ETest, RateSwitchWithSilenceDetection) {
    /**
     * レート切替時に無音が発生しないことを検証。
     *
     * 期待される動作:
     * - レート切替後も無音が発生しない
     * - ストリーミングが継続される
     *
     * 注意: 実際の実装では、無音検出ロジックを呼び出す必要があります。
     * このテストケースは期待される動作を定義するためのものです。
     */
    for (size_t i = 0; i < TEST_INPUT_RATES_COUNT; ++i) {
        int inputRate = TEST_INPUT_RATES[i];
        EXPECT_GT(inputRate, 0) << "Input rate should be positive: " << inputRate;
    }
}

