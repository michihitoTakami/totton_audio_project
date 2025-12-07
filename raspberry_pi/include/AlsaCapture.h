#pragma once

#include <string>

class AlsaCapture {
public:
    AlsaCapture();
    ~AlsaCapture();

    bool initialize(const std::string &deviceName);
    bool start();
    void stop();

private:
    std::string deviceName_;
};

