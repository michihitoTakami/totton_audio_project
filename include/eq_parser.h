#ifndef EQ_PARSER_H
#define EQ_PARSER_H

#include <string>
#include <vector>

namespace EQ {

// Filter types supported by Equalizer APO format
// Reference: https://sourceforge.net/p/equalizerapo/wiki/Configuration%20reference/
enum class FilterType : std::uint8_t {
    // Peaking filters
    PK,     // Peaking (parametric)
    MODAL,  // Modal filter (same as PK)
    PEQ,    // Parametric EQ (same as PK)
    // Pass filters
    LP,   // Low Pass
    LPQ,  // Low Pass with Q
    HP,   // High Pass
    HPQ,  // High Pass with Q
    BP,   // Band Pass
    // Notch and All-pass
    NO,  // Notch
    AP,  // All-pass
    // Shelf filters
    LS,   // Low Shelf
    HS,   // High Shelf
    LSC,  // Low Shelf Constant Q
    HSC,  // High Shelf Constant Q
    LSQ,  // Low Shelf with Q
    HSQ,  // High Shelf with Q
    // Fixed-slope shelf filters
    LS_6DB,   // Low Shelf 6dB/oct
    LS_12DB,  // Low Shelf 12dB/oct
    HS_6DB,   // High Shelf 6dB/oct
    HS_12DB,  // High Shelf 12dB/oct
};

// Single EQ band parameters
struct EqBand {
    bool enabled = true;
    FilterType type = FilterType::PK;
    double frequency = 1000.0;  // Center frequency (Hz)
    double gain = 0.0;          // Gain (dB)
    double q = 1.0;             // Q factor (bandwidth)
    bool hasBandwidthHz = false;
    double bandwidthHz = 0.0;  // Alternative bandwidth in Hz
    bool hasBandwidthOct = false;
    double bandwidthOct = 0.0;  // Alternative bandwidth in octaves
};

// Complete EQ profile
struct EqProfile {
    std::string name;
    double preampDb = 0.0;  // Preamp gain (dB)
    std::vector<EqBand> bands;

    bool isEmpty() const {
        return bands.empty() && preampDb == 0.0;
    }
    size_t activeBandCount() const;
};

// Parse Equalizer APO format file
// Returns true on success, false on parse error
bool parseEqFile(const std::string& filePath, EqProfile& profile);

// Parse from string content (for web API)
bool parseEqString(const std::string& content, EqProfile& profile);

// Get filter type name as string
const char* filterTypeName(FilterType type);

// Parse filter type from string
FilterType parseFilterType(const std::string& typeStr);

}  // namespace EQ

#endif  // EQ_PARSER_H
