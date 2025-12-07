#include "AlsaCapture.h"

#include <iostream>

AlsaCapture::AlsaCapture() = default;

AlsaCapture::~AlsaCapture() = default;

bool AlsaCapture::initialize(const std::string &deviceName)
{
    deviceName_ = deviceName;
    std::clog << "[AlsaCapture] initialize called for device: " << deviceName_
              << " (not implemented yet)" << std::endl;
    return false;
}

bool AlsaCapture::start()
{
    std::clog << "[AlsaCapture] start requested (not implemented yet)"
              << std::endl;
    return false;
}

void AlsaCapture::stop()
{
    std::clog << "[AlsaCapture] stop requested (not implemented yet)"
              << std::endl;
}

