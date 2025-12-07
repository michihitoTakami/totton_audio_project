#pragma once

#include <string>

enum class OutputDeviceType {
    Loopback,
    NullSink,
    Alsa,
};

struct OutputDeviceSpec {
    OutputDeviceType type{OutputDeviceType::Loopback};
    std::string alsaName{"hw:Loopback,0,0"};
    std::string userValue{"loopback"};

    std::string describe() const;
};

struct ParseOutputDeviceResult {
    bool ok{false};
    OutputDeviceSpec spec{};
    std::string error;
};

ParseOutputDeviceResult parseOutputDevice(const std::string &value);
