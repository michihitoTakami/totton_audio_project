/**
 * @file test_eq_parser.cpp
 * @brief Unit tests for EQ parser (Equalizer APO format)
 */

#include "eq_parser.h"

#include <gtest/gtest.h>

using namespace EQ;

class EqParserTest : public ::testing::Test {
   protected:
    void SetUp() override {}
};

// ============================================================
// filterTypeName tests
// ============================================================

TEST_F(EqParserTest, FilterTypeNamePK) {
    EXPECT_STREQ(filterTypeName(FilterType::PK), "PK");
}

TEST_F(EqParserTest, FilterTypeNameLS) {
    EXPECT_STREQ(filterTypeName(FilterType::LS), "LS");
}

TEST_F(EqParserTest, FilterTypeNameHS) {
    EXPECT_STREQ(filterTypeName(FilterType::HS), "HS");
}

TEST_F(EqParserTest, FilterTypeNameLP) {
    EXPECT_STREQ(filterTypeName(FilterType::LP), "LP");
}

TEST_F(EqParserTest, FilterTypeNameHP) {
    EXPECT_STREQ(filterTypeName(FilterType::HP), "HP");
}

// ============================================================
// parseFilterType tests
// ============================================================

TEST_F(EqParserTest, ParseFilterTypePK) {
    EXPECT_EQ(parseFilterType("PK"), FilterType::PK);
    EXPECT_EQ(parseFilterType("pk"), FilterType::PK);
    EXPECT_EQ(parseFilterType("PEAK"), FilterType::PK);
    EXPECT_EQ(parseFilterType("peaking"), FilterType::PK);
}

TEST_F(EqParserTest, ParseFilterTypeLS) {
    EXPECT_EQ(parseFilterType("LS"), FilterType::LS);
    EXPECT_EQ(parseFilterType("ls"), FilterType::LS);
    EXPECT_EQ(parseFilterType("LSC"), FilterType::LS);
    EXPECT_EQ(parseFilterType("LOWSHELF"), FilterType::LS);
}

TEST_F(EqParserTest, ParseFilterTypeHS) {
    EXPECT_EQ(parseFilterType("HS"), FilterType::HS);
    EXPECT_EQ(parseFilterType("hs"), FilterType::HS);
    EXPECT_EQ(parseFilterType("HSC"), FilterType::HS);
    EXPECT_EQ(parseFilterType("HIGHSHELF"), FilterType::HS);
}

TEST_F(EqParserTest, ParseFilterTypeLP) {
    EXPECT_EQ(parseFilterType("LP"), FilterType::LP);
    EXPECT_EQ(parseFilterType("LPQ"), FilterType::LP);
    EXPECT_EQ(parseFilterType("LOWPASS"), FilterType::LP);
}

TEST_F(EqParserTest, ParseFilterTypeHP) {
    EXPECT_EQ(parseFilterType("HP"), FilterType::HP);
    EXPECT_EQ(parseFilterType("HPQ"), FilterType::HP);
    EXPECT_EQ(parseFilterType("HIGHPASS"), FilterType::HP);
}

TEST_F(EqParserTest, ParseFilterTypeUnknownDefaultsToPK) {
    EXPECT_EQ(parseFilterType("UNKNOWN"), FilterType::PK);
    EXPECT_EQ(parseFilterType(""), FilterType::PK);
}

// ============================================================
// EqProfile tests
// ============================================================

TEST_F(EqParserTest, EqProfileIsEmptyWhenNoBands) {
    EqProfile profile;
    EXPECT_TRUE(profile.isEmpty());
}

TEST_F(EqParserTest, EqProfileIsEmptyWithPreampOnly) {
    EqProfile profile;
    profile.preampDb = -5.0;
    EXPECT_FALSE(profile.isEmpty());
}

TEST_F(EqParserTest, EqProfileIsNotEmptyWithBands) {
    EqProfile profile;
    profile.bands.push_back(EqBand{});
    EXPECT_FALSE(profile.isEmpty());
}

TEST_F(EqParserTest, EqProfileActiveBandCount) {
    EqProfile profile;

    // Add 3 bands: 2 enabled, 1 disabled
    EqBand band1;
    band1.enabled = true;
    profile.bands.push_back(band1);

    EqBand band2;
    band2.enabled = false;
    profile.bands.push_back(band2);

    EqBand band3;
    band3.enabled = true;
    profile.bands.push_back(band3);

    EXPECT_EQ(profile.activeBandCount(), 2u);
}

// ============================================================
// parseEqString tests
// ============================================================

TEST_F(EqParserTest, ParseEqStringPreampOnly) {
    EqProfile profile;
    std::string content = "Preamp: -5.5 dB";

    EXPECT_TRUE(parseEqString(content, profile));
    EXPECT_DOUBLE_EQ(profile.preampDb, -5.5);
    EXPECT_TRUE(profile.bands.empty());
}

TEST_F(EqParserTest, ParseEqStringSingleFilter) {
    EqProfile profile;
    std::string content = "Filter 1: ON PK Fc 1000 Hz Gain -3 dB Q 1.41";

    EXPECT_TRUE(parseEqString(content, profile));
    EXPECT_EQ(profile.bands.size(), 1u);
    EXPECT_TRUE(profile.bands[0].enabled);
    EXPECT_EQ(profile.bands[0].type, FilterType::PK);
    EXPECT_DOUBLE_EQ(profile.bands[0].frequency, 1000.0);
    EXPECT_DOUBLE_EQ(profile.bands[0].gain, -3.0);
    EXPECT_DOUBLE_EQ(profile.bands[0].q, 1.41);
}

TEST_F(EqParserTest, ParseEqStringDisabledFilter) {
    EqProfile profile;
    std::string content = "Filter 1: OFF PK Fc 1000 Hz Gain -3 dB Q 1.41";

    EXPECT_TRUE(parseEqString(content, profile));
    EXPECT_EQ(profile.bands.size(), 1u);
    EXPECT_FALSE(profile.bands[0].enabled);
}

TEST_F(EqParserTest, ParseEqStringMultipleFilters) {
    EqProfile profile;
    std::string content = R"(
Preamp: -6 dB
Filter 1: ON PK Fc 100 Hz Gain 3 dB Q 0.7
Filter 2: ON LS Fc 50 Hz Gain 2 dB Q 0.71
Filter 3: OFF HS Fc 10000 Hz Gain -2 dB Q 0.71
)";

    EXPECT_TRUE(parseEqString(content, profile));
    EXPECT_DOUBLE_EQ(profile.preampDb, -6.0);
    EXPECT_EQ(profile.bands.size(), 3u);
    EXPECT_EQ(profile.activeBandCount(), 2u);

    EXPECT_EQ(profile.bands[0].type, FilterType::PK);
    EXPECT_EQ(profile.bands[1].type, FilterType::LS);
    EXPECT_EQ(profile.bands[2].type, FilterType::HS);
}

TEST_F(EqParserTest, ParseEqStringSkipsComments) {
    EqProfile profile;
    std::string content = R"(
# This is a comment
; This is also a comment
Preamp: -3 dB
# Another comment
Filter 1: ON PK Fc 1000 Hz Gain -2 dB Q 1.0
)";

    EXPECT_TRUE(parseEqString(content, profile));
    EXPECT_DOUBLE_EQ(profile.preampDb, -3.0);
    EXPECT_EQ(profile.bands.size(), 1u);
}

TEST_F(EqParserTest, ParseEqStringEmptyContent) {
    EqProfile profile;
    std::string content = "";

    EXPECT_FALSE(parseEqString(content, profile));
}

TEST_F(EqParserTest, ParseEqStringOnlyComments) {
    EqProfile profile;
    std::string content = R"(
# Comment 1
; Comment 2
)";

    EXPECT_FALSE(parseEqString(content, profile));
}

TEST_F(EqParserTest, ParseEqStringDecimalFrequency) {
    EqProfile profile;
    std::string content = "Filter 1: ON PK Fc 140.3 Hz Gain -2.5 dB Q 0.81";

    EXPECT_TRUE(parseEqString(content, profile));
    EXPECT_DOUBLE_EQ(profile.bands[0].frequency, 140.3);
    EXPECT_DOUBLE_EQ(profile.bands[0].gain, -2.5);
    EXPECT_DOUBLE_EQ(profile.bands[0].q, 0.81);
}

TEST_F(EqParserTest, ParseEqStringPositiveGain) {
    EqProfile profile;
    std::string content = "Filter 1: ON PK Fc 1000 Hz Gain +3 dB Q 1.0";

    EXPECT_TRUE(parseEqString(content, profile));
    EXPECT_DOUBLE_EQ(profile.bands[0].gain, 3.0);
}

// ============================================================
// EqBand default values tests
// ============================================================

TEST_F(EqParserTest, EqBandDefaultValues) {
    EqBand band;
    EXPECT_TRUE(band.enabled);
    EXPECT_EQ(band.type, FilterType::PK);
    EXPECT_DOUBLE_EQ(band.frequency, 1000.0);
    EXPECT_DOUBLE_EQ(band.gain, 0.0);
    EXPECT_DOUBLE_EQ(band.q, 1.0);
}

// ============================================================
// New filter types tests (Issue #293)
// ============================================================

TEST_F(EqParserTest, ParseFilterTypeModal) {
    EXPECT_EQ(parseFilterType("MODAL"), FilterType::MODAL);
    EXPECT_EQ(parseFilterType("modal"), FilterType::MODAL);
}

TEST_F(EqParserTest, ParseFilterTypePEQ) {
    EXPECT_EQ(parseFilterType("PEQ"), FilterType::PEQ);
    EXPECT_EQ(parseFilterType("peq"), FilterType::PEQ);
}

TEST_F(EqParserTest, ParseFilterTypeLPQ) {
    EXPECT_EQ(parseFilterType("LPQ"), FilterType::LPQ);
    EXPECT_EQ(parseFilterType("lpq"), FilterType::LPQ);
}

TEST_F(EqParserTest, ParseFilterTypeHPQ) {
    EXPECT_EQ(parseFilterType("HPQ"), FilterType::HPQ);
    EXPECT_EQ(parseFilterType("hpq"), FilterType::HPQ);
}

TEST_F(EqParserTest, ParseFilterTypeBP) {
    EXPECT_EQ(parseFilterType("BP"), FilterType::BP);
    EXPECT_EQ(parseFilterType("BANDPASS"), FilterType::BP);
}

TEST_F(EqParserTest, ParseFilterTypeNO) {
    EXPECT_EQ(parseFilterType("NO"), FilterType::NO);
    EXPECT_EQ(parseFilterType("NOTCH"), FilterType::NO);
}

TEST_F(EqParserTest, ParseFilterTypeAP) {
    EXPECT_EQ(parseFilterType("AP"), FilterType::AP);
    EXPECT_EQ(parseFilterType("ALLPASS"), FilterType::AP);
}

TEST_F(EqParserTest, ParseFilterTypeLSC) {
    EXPECT_EQ(parseFilterType("LSC"), FilterType::LSC);
}

TEST_F(EqParserTest, ParseFilterTypeHSC) {
    EXPECT_EQ(parseFilterType("HSC"), FilterType::HSC);
}

TEST_F(EqParserTest, ParseFilterTypeLSQ) {
    EXPECT_EQ(parseFilterType("LSQ"), FilterType::LSQ);
}

TEST_F(EqParserTest, ParseFilterTypeHSQ) {
    EXPECT_EQ(parseFilterType("HSQ"), FilterType::HSQ);
}

TEST_F(EqParserTest, ParseFilterTypeLS6DB) {
    EXPECT_EQ(parseFilterType("LS 6DB"), FilterType::LS_6DB);
    EXPECT_EQ(parseFilterType("LS6DB"), FilterType::LS_6DB);
}

TEST_F(EqParserTest, ParseFilterTypeLS12DB) {
    EXPECT_EQ(parseFilterType("LS 12DB"), FilterType::LS_12DB);
    EXPECT_EQ(parseFilterType("LS12DB"), FilterType::LS_12DB);
}

TEST_F(EqParserTest, ParseFilterTypeHS6DB) {
    EXPECT_EQ(parseFilterType("HS 6DB"), FilterType::HS_6DB);
    EXPECT_EQ(parseFilterType("HS6DB"), FilterType::HS_6DB);
}

TEST_F(EqParserTest, ParseFilterTypeHS12DB) {
    EXPECT_EQ(parseFilterType("HS 12DB"), FilterType::HS_12DB);
    EXPECT_EQ(parseFilterType("HS12DB"), FilterType::HS_12DB);
}

// ============================================================
// Optional parameter tests (Issue #293)
// ============================================================

TEST_F(EqParserTest, ParseEqStringOptionalGain) {
    EqProfile profile;
    std::string content = "Filter 1: ON LP Fc 15000 Hz";

    EXPECT_TRUE(parseEqString(content, profile));
    EXPECT_EQ(profile.bands.size(), 1u);
    EXPECT_EQ(profile.bands[0].type, FilterType::LP);
    EXPECT_DOUBLE_EQ(profile.bands[0].frequency, 15000.0);
    EXPECT_DOUBLE_EQ(profile.bands[0].gain, 0.0);  // Default
    EXPECT_DOUBLE_EQ(profile.bands[0].q, 1.0);     // Default
}

TEST_F(EqParserTest, ParseEqStringOptionalQ) {
    EqProfile profile;
    std::string content = "Filter 1: ON HP Fc 25 Hz";

    EXPECT_TRUE(parseEqString(content, profile));
    EXPECT_EQ(profile.bands.size(), 1u);
    EXPECT_EQ(profile.bands[0].type, FilterType::HP);
    EXPECT_DOUBLE_EQ(profile.bands[0].frequency, 25.0);
    EXPECT_DOUBLE_EQ(profile.bands[0].gain, 0.0);  // Default
    EXPECT_DOUBLE_EQ(profile.bands[0].q, 1.0);     // Default
}

TEST_F(EqParserTest, ParseEqStringWithQButNoGain) {
    EqProfile profile;
    std::string content = "Filter 1: ON LPQ Fc 10000 Hz Q 0.707";

    EXPECT_TRUE(parseEqString(content, profile));
    EXPECT_EQ(profile.bands.size(), 1u);
    EXPECT_EQ(profile.bands[0].type, FilterType::LPQ);
    EXPECT_DOUBLE_EQ(profile.bands[0].frequency, 10000.0);
    EXPECT_DOUBLE_EQ(profile.bands[0].gain, 0.0);  // Default
    EXPECT_DOUBLE_EQ(profile.bands[0].q, 0.707);
}

TEST_F(EqParserTest, ParseEqStringNewFilterTypes) {
    EqProfile profile;
    std::string content = R"(
Preamp: -6 dB
Filter 1: ON MODAL Fc 100 Hz Gain -2.0 dB Q 1.0
Filter 2: ON PEQ Fc 200 Hz Gain 3.0 dB Q 1.5
Filter 3: ON LPQ Fc 10000 Hz Q 0.7
Filter 4: ON HPQ Fc 20 Hz Q 0.7
Filter 5: ON BP Fc 1000 Hz Q 2.0
Filter 6: ON NO Fc 500 Hz Q 5.0
Filter 7: ON AP Fc 2000 Hz Gain 0 dB Q 1.0
Filter 8: ON LS 6DB Fc 200 Hz Gain 4.0 dB
)";

    EXPECT_TRUE(parseEqString(content, profile));
    EXPECT_DOUBLE_EQ(profile.preampDb, -6.0);
    EXPECT_EQ(profile.bands.size(), 8u);
    EXPECT_EQ(profile.bands[0].type, FilterType::MODAL);
    EXPECT_EQ(profile.bands[1].type, FilterType::PEQ);
    EXPECT_EQ(profile.bands[2].type, FilterType::LPQ);
    EXPECT_EQ(profile.bands[3].type, FilterType::HPQ);
    EXPECT_EQ(profile.bands[4].type, FilterType::BP);
    EXPECT_EQ(profile.bands[5].type, FilterType::NO);
    EXPECT_EQ(profile.bands[6].type, FilterType::AP);
    EXPECT_EQ(profile.bands[7].type, FilterType::LS_6DB);
}
