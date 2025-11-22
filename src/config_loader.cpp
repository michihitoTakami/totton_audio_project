#include "config_loader.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

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
