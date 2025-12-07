#include "AlsaCapture.h"
#include "Options.h"
#include "TcpClient.h"

#include <alsa/asoundlib.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> gStopRequested{false};

void handleSignal(int /*signum*/) {
    gStopRequested.store(true);
}

std::uint16_t toPcmFormatCode(AlsaCapture::SampleFormat format) {
    switch (format) {
    case AlsaCapture::SampleFormat::S16_LE:
        return 1;
    case AlsaCapture::SampleFormat::S24_3LE:
        return 2;
    case AlsaCapture::SampleFormat::S32_LE:
        return 4;
    }
    return 0;
}

std::optional<std::string> selectCaptureDevice(const std::string &requested) {
    if (requested != "auto") {
        return requested;
    }

    void **hints = nullptr;
    if (snd_device_name_hint(-1, "pcm", &hints) != 0 || hints == nullptr) {
        return std::nullopt;
    }

    std::optional<std::string> found;
    for (void **hint = hints; *hint != nullptr; ++hint) {
        const char *name = snd_device_name_get_hint(*hint, "NAME");
        const char *ioid = snd_device_name_get_hint(*hint, "IOID");
        if (!name) {
            continue;
        }
        // Prefer input-capable devices.
        if (ioid && std::string{ioid} != "Input") {
            continue;
        }
        found = std::string{name};
        break;
    }
    snd_device_name_free_hint(hints);
    return found;
}

bool openCaptureWithRetry(AlsaCapture &capture, AlsaCapture::Config &cfg) {
    std::chrono::milliseconds backoff{1000};
    const std::chrono::milliseconds backoffMax{8000};
    while (!gStopRequested.load()) {
        if (capture.open(cfg) && capture.start()) {
            if (auto currentRate = capture.currentSampleRate()) {
                cfg.sampleRate = *currentRate;
            }
            return true;
        }
        logWarn("[rpi_pcm_bridge] Failed to open/start ALSA device, retrying in " +
                std::to_string(backoff.count()) + " ms");
        capture.close();
        std::this_thread::sleep_for(backoff);
        backoff = std::min(backoff * 2, backoffMax);
    }
    return false;
}

}  // namespace

int main(int argc, char **argv) {
    const std::string_view programName =
        (argc > 0 && argv[0] != nullptr) ? std::string_view{argv[0]} : "rpi_pcm_bridge";

    auto parsed = parseOptions(argc, argv, programName);
    if (parsed.showHelp) {
        return EXIT_SUCCESS;
    }
    if (parsed.showVersion) {
        return EXIT_SUCCESS;
    }
    if (parsed.hasError) {
        logError(parsed.errorMessage);
        return EXIT_FAILURE;
    }
    if (!parsed.options) {
        return EXIT_FAILURE;
    }
    const Options opt = *parsed.options;

    std::string resolvedDevice = opt.device;
    if (auto dev = selectCaptureDevice(opt.device)) {
        resolvedDevice = *dev;
    } else {
        logWarn("[rpi_pcm_bridge] auto device selection failed, using requested: " + opt.device);
    }

    setLogLevel(opt.logLevel);
    logInfo("[rpi_pcm_bridge] start device=" + resolvedDevice + " host=" + opt.host +
            " port=" + std::to_string(opt.port) + " rate=" + std::to_string(opt.rate) + " format=" +
            std::to_string(static_cast<int>(opt.format)) + " frames=" + std::to_string(opt.frames));

    struct sigaction sa {};
    sa.sa_handler = handleSignal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    AlsaCapture capture;
    AlsaCapture::Config cfg;
    cfg.deviceName = resolvedDevice;
    cfg.sampleRate = opt.rate;
    cfg.channels = 2;
    cfg.format = opt.format;
    cfg.periodFrames = opt.frames;

    if (!openCaptureWithRetry(capture, cfg)) {
        return EXIT_FAILURE;
    }

    PcmHeader header;
    header.sampleRate = cfg.sampleRate;
    header.channels = static_cast<std::uint16_t>(cfg.channels);
    header.format = toPcmFormatCode(cfg.format);
    unsigned int lastSampleRate = header.sampleRate;
    unsigned int lastChannels = cfg.channels;
    AlsaCapture::SampleFormat lastFormat = cfg.format;

    TcpClient client;
    if (!client.configure(opt.host, opt.port, header)) {
        logError("[rpi_pcm_bridge] Failed to connect TCP client");
        capture.stop();
        capture.close();
        return EXIT_FAILURE;
    }

    std::vector<std::uint8_t> buffer;
    bool success = true;
    int iteration = 0;
    while (!gStopRequested.load()) {
        int bytes = capture.read(buffer);
        if (bytes == -EPIPE) {
            logWarn("[rpi_pcm_bridge] XRUN recovered, continuing");
            continue;
        }
        if (bytes < 0) {
            logWarn("[rpi_pcm_bridge] Read failed: " + std::to_string(bytes) +
                    ". Reopening ALSA device and resending header.");
            capture.stop();
            capture.close();
            client.disconnect();

            if (!openCaptureWithRetry(capture, cfg)) {
                success = false;
                break;
            }
            header.sampleRate = cfg.sampleRate;
            lastSampleRate = header.sampleRate;
            if (!client.configure(opt.host, opt.port, header)) {
                logError("[rpi_pcm_bridge] Reconnect after capture reopen failed");
                success = false;
                break;
            }
            continue;
        }
        if (bytes == 0) {
            continue;
        }

        buffer.resize(static_cast<std::size_t>(bytes));
        if (!client.sendPcmChunk(buffer)) {
            logError("[rpi_pcm_bridge] Failed to send PCM chunk");
            success = false;
            break;
        }

        auto currentRate = capture.currentSampleRate();
        auto currentCh = capture.currentChannels();
        auto currentFmt = capture.currentFormat();
        if (currentRate && currentCh && currentFmt) {
            bool needRestart = false;
            if (*currentRate != lastSampleRate) {
                needRestart = true;
            }
            if (*currentCh != lastChannels) {
                needRestart = true;
            }
            if (*currentFmt != lastFormat) {
                needRestart = true;
            }
            if (needRestart) {
                logInfo("[rpi_pcm_bridge] Device params changed -> rate=" +
                        std::to_string(currentRate.value()) +
                        " ch=" + std::to_string(currentCh.value()) +
                        " fmt=" + std::to_string(static_cast<int>(currentFmt.value())) +
                        ". Re-opening capture and resending header.");
                capture.stop();
                capture.close();
                client.disconnect();

                cfg.sampleRate = currentRate.value();
                cfg.channels = currentCh.value();
                cfg.format = currentFmt.value();
                cfg.periodFrames = opt.frames;

                if (!openCaptureWithRetry(capture, cfg)) {
                    success = false;
                    break;
                }
                header.sampleRate = cfg.sampleRate;
                header.channels = static_cast<std::uint16_t>(cfg.channels);
                header.format = toPcmFormatCode(cfg.format);
                lastSampleRate = cfg.sampleRate;
                lastChannels = cfg.channels;
                lastFormat = cfg.format;

                if (!client.configure(opt.host, opt.port, header)) {
                    logError("[rpi_pcm_bridge] Reconnect after device param change failed");
                    success = false;
                    break;
                }
                continue;
            }
        }

        if (opt.iterations > 0 && ++iteration >= opt.iterations) {
            logInfo("[rpi_pcm_bridge] Iteration limit reached, stopping");
            break;
        }
    }

    if (gStopRequested.load()) {
        logInfo("[rpi_pcm_bridge] Signal received, shutting down");
    }

    capture.stop();
    capture.close();
    client.disconnect();

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
