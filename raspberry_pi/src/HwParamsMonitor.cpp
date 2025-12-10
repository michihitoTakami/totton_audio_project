// Utility to read ALSA hw_params from /proc for rate/format detection
#include "HwParamsMonitor.h"

#include "logging.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

// Docker環境では /proc/asound を /host_proc_asound にマウントする
const char *getProcAsoundPath() {
    // /host_proc_asound が存在すればそちらを使用（Docker環境）
    std::error_code ec;
    if (std::filesystem::exists("/host_proc_asound", ec) && !ec) {
        return "/host_proc_asound";
    }
    // 通常環境
    return "/proc/asound";
}

const std::array<unsigned int, 10> kAllowedRates = {44100,  48000,  88200,  96000,  176400,
                                                    192000, 352800, 384000, 705600, 768000};

std::optional<AlsaCapture::SampleFormat> parseFormatName(const std::string &value) {
    if (value == "S16_LE") {
        return AlsaCapture::SampleFormat::S16_LE;
    }
    if (value == "S24_3LE" || value == "S24_LE") {
        return AlsaCapture::SampleFormat::S24_3LE;
    }
    if (value == "S32_LE") {
        return AlsaCapture::SampleFormat::S32_LE;
    }
    return std::nullopt;
}

bool isAllowedRate(unsigned int rate) {
    return std::find(kAllowedRates.begin(), kAllowedRates.end(), rate) != kAllowedRates.end();
}

}  // namespace

HwParamsMonitor::HwParamsMonitor(std::string deviceName, std::string overridePath)
    : deviceName_(std::move(deviceName)), pathOverride_(std::move(overridePath)) {}

std::optional<std::string> HwParamsMonitor::resolveHwParamsPath() const {
    if (!pathOverride_.empty()) {
        return pathOverride_;
    }

    const auto colonPos = deviceName_.find(':');
    if (colonPos == std::string::npos || colonPos + 1 >= deviceName_.size()) {
        return std::nullopt;
    }
    const std::string afterPrefix = deviceName_.substr(colonPos + 1);
    const auto commaPos = afterPrefix.find(',');
    if (commaPos == std::string::npos || commaPos + 1 >= afterPrefix.size()) {
        // device名にカード/デバイス番号が含まれないケース (例: default, plughw)
        // -> /proc/asound 以下を走査して最初に見つかった capture hw_params を返す。
        std::error_code ec;
        const std::string procAsound = getProcAsoundPath();
        for (const auto &cardDir : std::filesystem::directory_iterator(procAsound, ec)) {
            if (ec) {
                break;
            }
            const auto name = cardDir.path().filename().string();
            if (name.rfind("card", 0) != 0 || !cardDir.is_directory()) {
                continue;
            }
            for (const auto &pcmDir : std::filesystem::directory_iterator(cardDir, ec)) {
                if (ec) {
                    break;
                }
                const auto pcmName = pcmDir.path().filename().string();
                if (pcmName.rfind("pcm", 0) != 0 || !pcmDir.is_directory()) {
                    continue;
                }
                const auto hwPath = pcmDir.path() / "sub0" / "hw_params";
                if (std::filesystem::exists(hwPath, ec)) {
                    return hwPath.string();
                }
            }
        }
        return std::nullopt;
    }
    const std::string cardStr = afterPrefix.substr(0, commaPos);
    const std::string devStr = afterPrefix.substr(commaPos + 1);

    for (char c : devStr) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return std::nullopt;
        }
    }

    const std::string procAsound = getProcAsoundPath();

    // cardStr が数字かどうかを判定
    bool cardIsNumeric =
        !cardStr.empty() &&
        std::all_of(cardStr.begin(), cardStr.end(),
                    [](unsigned char c) { return std::isdigit(c); });

    std::string cardPath;
    if (cardIsNumeric) {
        // 数字の場合: /proc/asound/card0 形式
        cardPath = procAsound + "/card" + cardStr;
    } else {
        // 名前の場合: /proc/asound/UAC2Gadget はシンボリックリンク
        cardPath = procAsound + "/" + cardStr;
    }

    const std::string primaryPath = cardPath + "/pcm" + devStr + "c/sub0/hw_params";
    std::error_code ec;
    if (std::filesystem::exists(primaryPath, ec)) {
        return primaryPath;
    }

    // sub0 が存在しない場合のフォールバック: sub1 など最初に見つかったものを返す
    const std::filesystem::path base = cardPath;
    for (const auto &pcmDir : std::filesystem::directory_iterator(base, ec)) {
        if (ec) {
            break;
        }
        const auto pcmName = pcmDir.path().filename().string();
        if (pcmName != "pcm" + devStr + "c" || !pcmDir.is_directory()) {
            continue;
        }
        for (const auto &subDir : std::filesystem::directory_iterator(pcmDir, ec)) {
            if (ec) {
                break;
            }
            if (subDir.path().filename().string().rfind("sub", 0) != 0) {
                continue;
            }
            const auto hwPath = subDir.path() / "hw_params";
            if (std::filesystem::exists(hwPath, ec)) {
                return hwPath.string();
            }
        }
    }

    return std::nullopt;
}

std::optional<CaptureParams> HwParamsMonitor::parseHwParams(const std::string &content) {
    std::istringstream ss(content);
    std::string line;
    unsigned int rate = 0;
    unsigned int channels = 0;
    std::optional<AlsaCapture::SampleFormat> fmt;

    while (std::getline(ss, line)) {
        if (line.rfind("rate:", 0) == 0) {
            std::istringstream rateStream(line.substr(5));
            rateStream >> rate;
        } else if (line.rfind("channels:", 0) == 0) {
            std::istringstream chStream(line.substr(9));
            chStream >> channels;
        } else if (line.rfind("format:", 0) == 0) {
            std::string value = line.substr(7);
            // Trim leading spaces
            value.erase(value.begin(),
                        std::find_if(value.begin(), value.end(),
                                     [](unsigned char c) { return !std::isspace(c); }));
            fmt = parseFormatName(value);
        }
    }

    if (rate == 0 || channels == 0 || !fmt) {
        return std::nullopt;
    }
    if (!isAllowedRate(rate)) {
        logWarn("[HwParamsMonitor] Ignoring unsupported rate in hw_params: " +
                std::to_string(rate));
        return std::nullopt;
    }
    CaptureParams params;
    params.sampleRate = rate;
    params.channels = channels;
    params.format = *fmt;
    return params;
}

std::optional<CaptureParams> HwParamsMonitor::readCurrent() const {
    auto path = resolveHwParamsPath();
    if (!path) {
        logError("[HwParamsMonitor] Could not resolve hw_params path for device " + deviceName_);
        return std::nullopt;
    }

    std::ifstream file(*path);
    if (!file.is_open()) {
        logError("[HwParamsMonitor] Failed to open hw_params at " + *path);
        return std::nullopt;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return parseHwParams(buffer.str());
}

std::string HwParamsMonitor::describe() const {
    auto path = resolveHwParamsPath();
    if (!path) {
        return "hw_params path unresolved";
    }
    return *path;
}
