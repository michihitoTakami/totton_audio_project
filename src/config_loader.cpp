#include "config_loader.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

PhaseType parsePhaseType(const std::string& str) {
    if (str == "linear") {
        return PhaseType::Linear;
    }
    // Default to Minimum for "minimum" or any invalid value
    return PhaseType::Minimum;
}

const char* phaseTypeToString(PhaseType type) {
    switch (type) {
    case PhaseType::Linear:
        return "linear";
    case PhaseType::Minimum:
    default:
        return "minimum";
    }
}

bool loadAppConfig(const std::filesystem::path& configPath, AppConfig& outConfig, bool verbose) {
    outConfig = AppConfig{};

    std::ifstream file(configPath);
    if (!file.is_open()) {
        if (verbose) {
            std::cout << "Config: " << configPath << " not found, using defaults" << std::endl;
        }
        return false;
    }

    try {
        nlohmann::json j;
        file >> j;

        if (j.contains("alsaDevice"))
            outConfig.alsaDevice = j["alsaDevice"].get<std::string>();
        if (j.contains("bufferSize"))
            outConfig.bufferSize = j["bufferSize"].get<int>();
        if (j.contains("periodSize"))
            outConfig.periodSize = j["periodSize"].get<int>();
        if (j.contains("upsampleRatio"))
            outConfig.upsampleRatio = j["upsampleRatio"].get<int>();
        if (j.contains("blockSize"))
            outConfig.blockSize = j["blockSize"].get<int>();
        if (j.contains("gain"))
            outConfig.gain = j["gain"].get<float>();
        if (j.contains("filterPath"))
            outConfig.filterPath = j["filterPath"].get<std::string>();
        if (j.contains("inputSampleRate"))
            outConfig.inputSampleRate = j["inputSampleRate"].get<int>();
        if (j.contains("phaseType"))
            outConfig.phaseType = parsePhaseType(j["phaseType"].get<std::string>());

        // Quad-phase mode settings
        if (j.contains("quadPhaseEnabled"))
            outConfig.quadPhaseEnabled = j["quadPhaseEnabled"].get<bool>();
        if (j.contains("filterPath44kMin"))
            outConfig.filterPath44kMin = j["filterPath44kMin"].get<std::string>();
        if (j.contains("filterPath48kMin"))
            outConfig.filterPath48kMin = j["filterPath48kMin"].get<std::string>();
        if (j.contains("filterPath44kLinear"))
            outConfig.filterPath44kLinear = j["filterPath44kLinear"].get<std::string>();
        if (j.contains("filterPath48kLinear"))
            outConfig.filterPath48kLinear = j["filterPath48kLinear"].get<std::string>();

        // EQ settings
        if (j.contains("eqEnabled"))
            outConfig.eqEnabled = j["eqEnabled"].get<bool>();
        if (j.contains("eqProfilePath"))
            outConfig.eqProfilePath = j["eqProfilePath"].get<std::string>();

        if (verbose) {
            std::cout << "Config: Loaded from " << std::filesystem::absolute(configPath)
                      << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        if (verbose) {
            std::cerr << "Config: Failed to parse " << configPath << ": " << e.what() << std::endl;
        }
        return false;
    }
}
