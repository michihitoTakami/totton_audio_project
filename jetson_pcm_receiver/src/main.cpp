#include <cstdlib>
#include <iostream>
#include <string>

#include "alsa_playback.h"
#include "pcm_stream_handler.h"
#include "tcp_server.h"

namespace {

struct AppOptions {
    int port = 46001;
    std::string device = "hw:Loopback,0,0";
};

void printHelp(const char *exeName) {
    std::cout << "jetson-pcm-receiver (scaffold)\n";
    std::cout << "Usage: " << exeName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -p, --port <number>     TCP listen port (default: 46001)\n";
    std::cout << "  -d, --device <name>     ALSA playback device (default: hw:Loopback,0,0)\n";
    std::cout << "  -h, --help              Show this help\n";
    std::cout << std::endl;
}

bool parseArgs(int argc, char **argv, AppOptions &options) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            printHelp(argv[0]);
            return false;
        }
        if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            options.port = std::atoi(argv[++i]);
            continue;
        }
        if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            options.device = argv[++i];
            continue;
        }

        std::cerr << "未知の引数: " << arg << std::endl;
        printHelp(argv[0]);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    AppOptions options;
    if (!parseArgs(argc, argv, options)) {
        return 1;
    }

    std::cout << "[jetson-pcm-receiver] 雛形ビルド - 実装は未着手です" << std::endl;
    std::cout << "  - port:   " << options.port << std::endl;
    std::cout << "  - device: " << options.device << std::endl;

    TcpServer server(options.port);
    AlsaPlayback playback(options.device);
    PcmStreamHandler handler(playback, server);

    server.start();
    handler.run();
    server.stop();

    std::cout << "[jetson-pcm-receiver] 終了" << std::endl;
    return 0;
}

