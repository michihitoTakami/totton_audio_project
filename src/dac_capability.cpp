#include "dac_capability.h"

#include <algorithm>
#include <alsa/asoundlib.h>
#include <iostream>

namespace DacCapability {

Capability scan(const std::string& device) {
    Capability cap;
    cap.deviceName = device;
    cap.isValid = false;
    cap.maxSampleRate = 0;
    cap.minSampleRate = 0;
    cap.maxChannels = 0;

    snd_pcm_t* pcm = nullptr;
    snd_pcm_hw_params_t* params = nullptr;

    int err = snd_pcm_open(&pcm, device.c_str(), SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK);
    if (err < 0) {
        cap.errorMessage = "Cannot open device: " + std::string(snd_strerror(err));
        return cap;
    }

    snd_pcm_hw_params_alloca(&params);

    err = snd_pcm_hw_params_any(pcm, params);
    if (err < 0) {
        cap.errorMessage = "Cannot get hardware params: " + std::string(snd_strerror(err));
        snd_pcm_close(pcm);
        return cap;
    }

    unsigned int minRate, maxRate;
    int dir;

    err = snd_pcm_hw_params_get_rate_min(params, &minRate, &dir);
    if (err < 0) {
        cap.errorMessage = "Cannot get min rate: " + std::string(snd_strerror(err));
        snd_pcm_close(pcm);
        return cap;
    }

    err = snd_pcm_hw_params_get_rate_max(params, &maxRate, &dir);
    if (err < 0) {
        cap.errorMessage = "Cannot get max rate: " + std::string(snd_strerror(err));
        snd_pcm_close(pcm);
        return cap;
    }

    cap.minSampleRate = static_cast<int>(minRate);
    cap.maxSampleRate = static_cast<int>(maxRate);

    unsigned int maxChannels;
    err = snd_pcm_hw_params_get_channels_max(params, &maxChannels);
    if (err >= 0) {
        cap.maxChannels = static_cast<int>(maxChannels);
    }

    const int testRates[] = {44100,  48000,  88200,  96000,  176400,  192000,
                             352800, 384000, 705600, 768000, 1411200, 1536000};

    for (int rate : testRates) {
        if (rate >= cap.minSampleRate && rate <= cap.maxSampleRate) {
            unsigned int testRate = static_cast<unsigned int>(rate);
            err = snd_pcm_hw_params_test_rate(pcm, params, testRate, 0);
            if (err == 0) {
                cap.supportedRates.push_back(rate);
            }
        }
    }

    snd_pcm_close(pcm);
    cap.isValid = true;
    return cap;
}

std::vector<std::string> listPlaybackDevices() {
    std::vector<std::string> devices;
    devices.push_back("default");
    devices.push_back("hw:0");
    devices.push_back("plughw:0");

    int card = -1;
    while (snd_card_next(&card) >= 0 && card >= 0) {
        std::string hwDevice = "hw:" + std::to_string(card);
        if (std::find(devices.begin(), devices.end(), hwDevice) == devices.end()) {
            devices.push_back(hwDevice);
        }
    }
    return devices;
}

bool isRateSupported(const Capability& cap, int sampleRate) {
    if (!cap.isValid)
        return false;
    if (sampleRate < cap.minSampleRate || sampleRate > cap.maxSampleRate)
        return false;
    if (!cap.supportedRates.empty()) {
        return std::find(cap.supportedRates.begin(), cap.supportedRates.end(), sampleRate) !=
               cap.supportedRates.end();
    }
    return true;
}

int getBestSupportedRate(const Capability& cap, int requestedRate) {
    if (!cap.isValid)
        return 0;
    if (isRateSupported(cap, requestedRate))
        return requestedRate;

    int bestRate = 0;
    if (!cap.supportedRates.empty()) {
        for (int rate : cap.supportedRates) {
            if (rate <= requestedRate && rate > bestRate) {
                bestRate = rate;
            }
        }
    } else {
        bestRate = std::min(requestedRate, cap.maxSampleRate);
    }
    return bestRate;
}

}  // namespace DacCapability
