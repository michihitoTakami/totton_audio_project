#include "daemon/control/handlers/handler_registry.h"

namespace daemon_control::handlers {

HandlerRegistry::HandlerRegistry(HandlerRegistryDependencies deps) : deps_(deps) {}

void HandlerRegistry::registerDefaults() {
    if (deps_.dispatcher) {
        deps_.dispatcher->subscribe([](const daemon_core::api::RateChangeRequested&) {});
        deps_.dispatcher->subscribe([](const daemon_core::api::DeviceChangeRequested&) {});
        deps_.dispatcher->subscribe([](const daemon_core::api::FilterSwitchRequested&) {});
    }
    registered_ = 3;
}

size_t HandlerRegistry::registeredCount() const {
    return registered_;
}

}  // namespace daemon_control::handlers
