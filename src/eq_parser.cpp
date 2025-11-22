#include "eq_parser.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

namespace EQ {

size_t EqProfile::activeBandCount() const {
    size_t count = 0;
    for (const auto& band : bands) {
        if (band.enabled)
            ++count;
    }
    return count;
}

const char* filterTypeName(FilterType type) {
    switch (type) {
    case FilterType::PK:
        return "PK";
    case FilterType::LS:
        return "LS";
    case FilterType::HS:
        return "HS";
    case FilterType::LP:
        return "LP";
    case FilterType::HP:
        return "HP";
    default:
        return "??";
    }
}

FilterType parseFilterType(const std::string& typeStr) {
    std::string upper = typeStr;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);

    if (upper == "PK" || upper == "PEAK" || upper == "PEAKING")
        return FilterType::PK;
    if (upper == "LS" || upper == "LSC" || upper == "LOWSHELF")
        return FilterType::LS;
    if (upper == "HS" || upper == "HSC" || upper == "HIGHSHELF")
        return FilterType::HS;
    if (upper == "LP" || upper == "LPQ" || upper == "LOWPASS")
        return FilterType::LP;
    if (upper == "HP" || upper == "HPQ" || upper == "HIGHPASS")
        return FilterType::HP;

    return FilterType::PK;  // Default to peaking
}

// Helper to trim whitespace
static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
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

    if (num.empty() || num == "-" || num == "+")
        return 0.0;
    return std::stod(num);
}

bool parseEqString(const std::string& content, EqProfile& profile) {
    profile.bands.clear();
    profile.preampDb = 0.0;

    std::istringstream stream(content);
    std::string line;

    // Regex patterns for parsing
    // Preamp: -10.5db or Preamp: -10.5 dB
    std::regex preampRegex(R"(Preamp:\s*([-+]?\d+\.?\d*)\s*[dD][bB]?)", std::regex::icase);

    // Filter N: ON/OFF TYPE Fc FREQ Hz Gain GAIN dB Q QVAL
    // Example: Filter 1: ON PK Fc 140.3 Hz Gain -2 dB Q 0.81
    std::regex filterRegex(
        R"(Filter\s+(\d+):\s*(ON|OFF)\s+(\w+)\s+Fc\s+([\d.]+)\s*Hz\s+Gain\s+([-+]?\d+\.?\d*)\s*dB\s+Q\s+([\d.]+))",
        std::regex::icase);

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

        // Try to match filter line
        if (std::regex_search(line, match, filterRegex)) {
            EqBand band;
            // match[1] = filter number (not used directly)
            band.enabled = (match[2].str() == "ON" || match[2].str() == "on");
            band.type = parseFilterType(match[3].str());
            band.frequency = std::stod(match[4].str());
            band.gain = std::stod(match[5].str());
            band.q = std::stod(match[6].str());

            profile.bands.push_back(band);
        }
    }

    return !profile.bands.empty() || profile.preampDb != 0.0;
}

bool parseEqFile(const std::string& filePath, EqProfile& profile) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "EQ Parser: Cannot open file: " << filePath << std::endl;
        return false;
    }

    // Extract profile name from filename
    size_t lastSlash = filePath.find_last_of("/\\");
    size_t lastDot = filePath.find_last_of('.');
    if (lastSlash == std::string::npos)
        lastSlash = 0;
    else
        lastSlash++;

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
                  << " active bands, preamp " << profile.preampDb << " dB" << std::endl;
    }

    return result;
}

}  // namespace EQ
