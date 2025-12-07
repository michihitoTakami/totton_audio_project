#include "AlsaCapture.h"
#include "Options.h"
#include "TcpClient.h"

#include <alsa/asoundlib.h>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::uint16_t toHeaderFormat(AlsaCapture::SampleFormat format) {
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
        std::cerr << parsed.errorMessage << '\n';
        return EXIT_FAILURE;
    }
    if (!parsed.options) {
        return EXIT_FAILURE;
    }
    const Options opt = *parsed.options;

    std::clog << "[rpi_pcm_bridge] start device=" << opt.device << " host=" << opt.host
              << " port=" << opt.port << " rate=" << opt.rate
              << " format=" << static_cast<int>(opt.format) << " frames=" << opt.frames
              << " log_level=" << opt.logLevel << '\n';

    AlsaCapture capture;
    AlsaCapture::Config cfg;
    cfg.deviceName = opt.device;
    cfg.sampleRate = opt.rate;
    cfg.channels = 2;
    cfg.format = opt.format;
    cfg.periodFrames = opt.frames;

    if (!capture.open(cfg)) {
        std::cerr << "[rpi_pcm_bridge] Failed to open device" << '\n';
        return EXIT_FAILURE;
    }
    if (!capture.start()) {
        std::cerr << "[rpi_pcm_bridge] Failed to start capture" << '\n';
        return EXIT_FAILURE;
    }

    TcpClient client;
    PcmHeader header{};
    header.sampleRate = cfg.sampleRate;
    header.channels = cfg.channels;
    header.format = toHeaderFormat(cfg.format);
    if (!client.configure(opt.host, opt.port, header)) {
        std::cerr << "[rpi_pcm_bridge] Failed to connect to TCP server" << '\n';
        capture.stop();
        capture.close();
        return EXIT_FAILURE;
    }

    std::vector<std::uint8_t> buffer;
    bool success = true;
    int iteration = 0;
    while (opt.iterations < 0 || iteration < opt.iterations) {
        int bytes = capture.read(buffer);
        if (bytes == -EPIPE) {
            std::clog << "[rpi_pcm_bridge] XRUN recovered, continuing" << '\n';
            continue;
        }
        if (bytes < 0) {
            std::clog << "[rpi_pcm_bridge] Read failed: " << bytes << '\n';
            success = false;
            break;
        }
        buffer.resize(static_cast<std::size_t>(bytes));
        if (!client.sendPcmChunk(buffer)) {
            std::clog << "[rpi_pcm_bridge] Send failed, aborting" << '\n';
            success = false;
            break;
        }
        ++iteration;
    }

    capture.stop();
    capture.close();
    client.disconnect();

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
