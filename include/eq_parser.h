#ifndef EQ_PARSER_H
#define EQ_PARSER_H

#include <string>
#include <vector>

namespace EQ {

// Filter types supported by Equalizer APO format
enum class FilterType {
    PK,  // Peaking (parametric)
    LS,  // Low Shelf
    HS,  // High Shelf
    LP,  // Low Pass (future)
    HP,  // High Pass (future)
};

// Single EQ band parameters
struct EqBand {
    bool enabled = true;
    FilterType type = FilterType::PK;
    double frequency = 1000.0;  // Center frequency (Hz)
    double gain = 0.0;          // Gain (dB)
    double q = 1.0;             // Q factor (bandwidth)
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
