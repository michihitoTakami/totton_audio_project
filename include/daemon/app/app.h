#pragma once

#include <memory>
#include <string>

namespace daemon_app {

struct RuntimeState;

struct AppOptions {
    std::string configFilePath;
    std::string statsFilePath;
};

class App {
   public:
    App();
    ~App();

    int run(const AppOptions& options, int argc, char* argv[]);

   private:
    std::unique_ptr<RuntimeState> state_;
};

}  // namespace daemon_app
