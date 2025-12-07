#include "AlsaCapture.h"
#include "Options.h"
#include "TcpClient.h"

#include <alsa/asoundlib.h>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
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

    setLogLevel(opt.logLevel);
    logInfo("[rpi_pcm_bridge] start device=" + opt.device + " host=" + opt.host +
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
    cfg.deviceName = opt.device;
    cfg.sampleRate = opt.rate;
    cfg.channels = 2;
    cfg.format = opt.format;
    cfg.periodFrames = opt.frames;

    if (!capture.open(cfg)) {
        logError("[rpi_pcm_bridge] Failed to open device");
        return EXIT_FAILURE;
    }
    if (!capture.start()) {
        logError("[rpi_pcm_bridge] Failed to start capture");
        capture.close();
        return EXIT_FAILURE;
    }

    PcmHeader header;
    header.sampleRate = cfg.sampleRate;
    header.channels = static_cast<std::uint16_t>(cfg.channels);
    header.format = toPcmFormatCode(cfg.format);
    unsigned int lastSampleRate = header.sampleRate;

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
            logError("[rpi_pcm_bridge] Read failed: " + std::to_string(bytes));
            success = false;
            break;
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

        if (auto currentRate = capture.currentSampleRate()) {
            if (*currentRate != lastSampleRate) {
                lastSampleRate = *currentRate;
                header.sampleRate = *currentRate;
                logInfo("[rpi_pcm_bridge] Sample rate change detected -> " +
                        std::to_string(*currentRate) + " Hz. Re-sending header.");
                client.disconnect();
                if (!client.configure(opt.host, opt.port, header)) {
                    logError("[rpi_pcm_bridge] Reconnect after rate change failed");
                    success = false;
                    break;
                }
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
