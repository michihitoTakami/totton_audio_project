#pragma once

#include "daemon/app/runtime_state.h"

#include <optional>
#include <string>

namespace daemon_app {

struct AppOverrides {
    std::optional<std::string> alsaDevice;
    std::optional<std::string> filterPath;
};

class App {
   public:
    App(RuntimeState& state, std::string configFilePath, std::string statsFilePath);

    int run(const AppOverrides& overrides);

   private:
    RuntimeState& state_;
    std::string configFilePath_;
    std::string statsFilePath_;
};

}  // namespace daemon_app
