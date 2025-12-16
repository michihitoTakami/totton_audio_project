#include "audio/eq_parser.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

namespace EQ {

size_t EqProfile::activeBandCount() const {
    size_t count = 0;
    for (const auto& band : bands) {
        if (band.enabled) {
            ++count;
        }
    }
    return count;
}

const char* filterTypeName(FilterType type) {
    switch (type) {
    case FilterType::PK:
        return "PK";
    case FilterType::MODAL:
        return "MODAL";
    case FilterType::PEQ:
        return "PEQ";
    case FilterType::LP:
        return "LP";
    case FilterType::LPQ:
        return "LPQ";
    case FilterType::HP:
        return "HP";
    case FilterType::HPQ:
        return "HPQ";
    case FilterType::BP:
        return "BP";
    case FilterType::NO:
        return "NO";
    case FilterType::AP:
        return "AP";
    case FilterType::LS:
        return "LS";
    case FilterType::HS:
        return "HS";
    case FilterType::LSC:
        return "LSC";
    case FilterType::HSC:
        return "HSC";
    case FilterType::LSQ:
        return "LSQ";
    case FilterType::HSQ:
        return "HSQ";
    case FilterType::LS_6DB:
        return "LS 6dB";
    case FilterType::LS_12DB:
        return "LS 12dB";
    case FilterType::HS_6DB:
        return "HS 6dB";
    case FilterType::HS_12DB:
        return "HS 12dB";
    default:
        return "??";
    }
}

FilterType parseFilterType(const std::string& typeStr) {
    std::string upper = typeStr;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);

    // Peaking filters
    if (upper == "PK" || upper == "PEAK" || upper == "PEAKING") {
        return FilterType::PK;
    }
    if (upper == "MODAL") {
        return FilterType::MODAL;
    }
    if (upper == "PEQ") {
        return FilterType::PEQ;
    }

    // Pass filters
    if (upper == "LP" || upper == "LOWPASS") {
        return FilterType::LP;
    }
    if (upper == "LPQ") {
        return FilterType::LPQ;
    }
    if (upper == "HP" || upper == "HIGHPASS") {
        return FilterType::HP;
    }
    if (upper == "HPQ") {
        return FilterType::HPQ;
    }
    if (upper == "BP" || upper == "BANDPASS") {
        return FilterType::BP;
    }

    // Notch and All-pass
    if (upper == "NO" || upper == "NOTCH") {
        return FilterType::NO;
    }
    if (upper == "AP" || upper == "ALLPASS") {
        return FilterType::AP;
    }

    // Shelf filters
    if (upper == "LS" || upper == "LOWSHELF") {
        return FilterType::LS;
    }
    if (upper == "HS" || upper == "HIGHSHELF") {
        return FilterType::HS;
    }
    if (upper == "LSC") {
        return FilterType::LSC;
    }
    if (upper == "HSC") {
        return FilterType::HSC;
    }
    if (upper == "LSQ") {
        return FilterType::LSQ;
    }
    if (upper == "HSQ") {
        return FilterType::HSQ;
    }

    // Fixed-slope shelf filters (with space handling)
    if (upper == "LS 6DB" || upper == "LS6DB") {
        return FilterType::LS_6DB;
    }
    if (upper == "LS 12DB" || upper == "LS12DB") {
        return FilterType::LS_12DB;
    }
    if (upper == "HS 6DB" || upper == "HS6DB") {
        return FilterType::HS_6DB;
    }
    if (upper == "HS 12DB" || upper == "HS12DB") {
        return FilterType::HS_12DB;
    }

    return FilterType::PK;  // Default to peaking
}

// Helper to trim whitespace
static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Helper to extract number from string like "140.3 Hz" or "-2 dB"
static double extractNumber(const std::string& s) {
    std::string num;
    bool hasDecimal = false;
    bool hasSign = false;

    for (char c : s) {
        if (c == '-' || c == '+') {
            if (!hasSign && num.empty()) {
                num += c;
                hasSign = true;
            }
        } else if (c == '.') {
            if (!hasDecimal) {
                num += c;
                hasDecimal = true;
            }
        } else if (std::isdigit(c)) {
            num += c;
        } else if (!num.empty() && !std::isdigit(c) && c != '.') {
            break;  // Stop at first non-numeric after we have digits
        }
    }

    if (num.empty() || num == "-" || num == "+") {
        return 0.0;
    }
    return std::stod(num);
}

static double bandwidthOctToQ(double bandwidthOct) {
    constexpr double DEFAULT_Q = 1.0;
    constexpr double LN2_OVER2 = 0.34657359037935203;  // ln(2) / 2

    if (bandwidthOct <= 0.0) {
        return DEFAULT_Q;
    }

    double denom = 2.0 * std::sinh(LN2_OVER2 * bandwidthOct);
    if (denom <= 0.0) {
        return DEFAULT_Q;
    }

    return 1.0 / denom;
}

static double bandwidthHzToQ(double centerFrequency, double bandwidthHz) {
    constexpr double DEFAULT_Q = 1.0;

    if (centerFrequency <= 0.0 || bandwidthHz <= 0.0) {
        return DEFAULT_Q;
    }

    return centerFrequency / bandwidthHz;
}

bool parseEqString(const std::string& content, EqProfile& profile) {
    profile.bands.clear();
    profile.preampDb = 0.0;

    std::istringstream stream(content);
    std::string line;

    // Regex patterns for parsing
    // Preamp: -10.5db or Preamp: -10.5 dB
    std::regex preampRegex(R"(Preamp:\s*([-+]?\d+\.?\d*)\s*[dD][bB]?)", std::regex::icase);

    // Base filter pattern: Filter N: ON/OFF TYPE Fc FREQ [Hz]
    // Gain and Q are now optional and parsed separately. Filter numbers can be omitted.
    std::regex filterBaseRegex(
        R"(Filter\s*(\d+)?\s*:\s*(ON|OFF)\s+(.+?)\s+Fc\s+([\d.]+)\s*(?:Hz)?)", std::regex::icase);

    // Optional parameter patterns
    std::regex gainRegex(R"(Gain\s+([-+]?\d+\.?\d*)\s*dB)", std::regex::icase);
    std::regex qRegex(R"(Q\s+([\d.]+))", std::regex::icase);

    while (std::getline(stream, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;  // Skip empty lines and comments
        }

        std::smatch match;

        // Try to match preamp line
        if (std::regex_search(line, match, preampRegex)) {
            profile.preampDb = std::stod(match[1].str());
            continue;
        }

        // Try to match filter line with base pattern
        if (std::regex_search(line, match, filterBaseRegex)) {
            EqBand band;
            std::string state = match[2].str();
            std::transform(state.begin(), state.end(), state.begin(), ::toupper);
            band.enabled = (state == "ON");
            std::string typeStr = trim(match[3].str());
            band.type = parseFilterType(typeStr);
            band.frequency = std::stod(match[4].str());

            // Extract optional Gain parameter
            std::smatch gainMatch;
            if (std::regex_search(line, gainMatch, gainRegex)) {
                band.gain = std::stod(gainMatch[1].str());
            } else {
                band.gain = 0.0;  // Default gain if not specified
            }

            // Extract optional Q parameter
            std::smatch qMatch;
            bool qProvided = false;
            if (std::regex_search(line, qMatch, qRegex)) {
                band.q = std::stod(qMatch[1].str());
                qProvided = true;
            } else {
                band.q = 1.0;  // Default Q if not specified
            }

            std::regex bwOctRegex(R"(BW\s+Oct\s+([-+]?\d+\.?\d*))", std::regex::icase);
            std::smatch bwOctMatch;
            if (std::regex_search(line, bwOctMatch, bwOctRegex)) {
                band.hasBandwidthOct = true;
                band.bandwidthOct = std::stod(bwOctMatch[1].str());
                if (!qProvided) {
                    band.q = bandwidthOctToQ(band.bandwidthOct);
                }
            }

            std::regex bwRegex(R"(BW\s+([-+]?\d+\.?\d*)\s*(?:Hz)?)", std::regex::icase);
            std::smatch bwMatch;
            if (std::regex_search(line, bwMatch, bwRegex)) {
                band.hasBandwidthHz = true;
                band.bandwidthHz = std::stod(bwMatch[1].str());
                if (!qProvided && !band.hasBandwidthOct) {
                    band.q = bandwidthHzToQ(band.frequency, band.bandwidthHz);
                }
            }

            profile.bands.push_back(band);
        }
    }

    return !profile.bands.empty() || profile.preampDb != 0.0;
}

bool parseEqFile(const std::string& filePath, EqProfile& profile) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "EQ Parser: Cannot open file: " << filePath << '\n';
        return false;
    }

    // Extract profile name from filename
    size_t lastSlash = filePath.find_last_of("/\\");
    size_t lastDot = filePath.find_last_of('.');
    if (lastSlash == std::string::npos) {
        lastSlash = 0;
    } else {
        lastSlash++;
    }

    if (lastDot != std::string::npos && lastDot > lastSlash) {
        profile.name = filePath.substr(lastSlash, lastDot - lastSlash);
    } else {
        profile.name = filePath.substr(lastSlash);
    }

    // Read entire file
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    bool result = parseEqString(buffer.str(), profile);

    if (result) {
        std::cout << "EQ Parser: Loaded '" << profile.name << "' with " << profile.activeBandCount()
                  << " active bands, preamp " << profile.preampDb << " dB" << '\n';
    }

    return result;
}

}  // namespace EQ
