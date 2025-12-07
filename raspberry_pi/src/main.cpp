#include "AlsaCapture.h"
#include "TcpClient.h"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

void printHelp(std::string_view programName)
{
    std::cout << "Usage: " << programName << " [--help]" << std::endl
              << std::endl
              << "Prototype PCM bridge entrypoint for Raspberry Pi." << std::endl
              << "Functionality is not implemented yet; this binary currently"
              << " serves as a build test placeholder." << std::endl;
}

} // namespace

int main(int argc, char **argv)
{
    const std::string_view programName = (argc > 0 && argv[0] != nullptr)
        ? std::string_view{argv[0]}
        : "rpi_pcm_bridge";

    for(int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if(arg == "-h" || arg == "--help") {
            printHelp(programName);
            return EXIT_SUCCESS;
        }
    }

    std::cout << "[rpi_pcm_bridge] Prototype build - functionality not"
              << " implemented yet." << std::endl;

    AlsaCapture capture;
    TcpClient client;

    std::clog << "[rpi_pcm_bridge] Created AlsaCapture and TcpClient stubs."
              << std::endl;

    return EXIT_SUCCESS;
}

