#include "output_device.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

OutputDeviceSpec loopbackSpec(std::string userValue) {
    OutputDeviceSpec spec;
    spec.type = OutputDeviceType::Loopback;
    spec.alsaName = "hw:Loopback,0,0";
    spec.userValue = std::move(userValue);
    return spec;
}

OutputDeviceSpec nullSpec(std::string userValue) {
    OutputDeviceSpec spec;
    spec.type = OutputDeviceType::NullSink;
    spec.alsaName = "null";
    spec.userValue = std::move(userValue);
    return spec;
}

OutputDeviceSpec alsaSpec(std::string alsaName, std::string userValue) {
    OutputDeviceSpec spec;
    spec.type = OutputDeviceType::Alsa;
    spec.alsaName = std::move(alsaName);
    spec.userValue = std::move(userValue);
    return spec;
}

}  // namespace

std::string OutputDeviceSpec::describe() const {
    std::string kind;
    switch (type) {
    case OutputDeviceType::Loopback:
        kind = "loopback";
        break;
    case OutputDeviceType::NullSink:
        kind = "null";
        break;
    case OutputDeviceType::Alsa:
        kind = "alsa";
        break;
    }
    return kind + " (" + alsaName + ")";
}

ParseOutputDeviceResult parseOutputDevice(const std::string &value) {
    ParseOutputDeviceResult result;
    if (value.empty()) {
        result.error = "device value must not be empty";
        return result;
    }

    const std::string lower = toLower(value);
    if (lower == "loopback" || lower == "loopback-playback") {
        result.ok = true;
        result.spec = loopbackSpec(value);
        return result;
    }
    if (lower == "null") {
        result.ok = true;
        result.spec = nullSpec(value);
        return result;
    }

    constexpr std::string_view kAlsaPrefix = "alsa:";
    if (lower.rfind(kAlsaPrefix, 0) == 0) {
        std::string deviceName = value.substr(kAlsaPrefix.size());
        if (deviceName.empty()) {
            result.error = "alsa device name is empty (expected alsa:<pcm-name>)";
            return result;
        }
        result.ok = true;
        result.spec = alsaSpec(deviceName, value);
        return result;
    }

    // Treat any other value as a raw ALSA PCM name to keep backward compatibility.
    result.ok = true;
    result.spec = alsaSpec(value, value);
    return result;
}
