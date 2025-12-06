#pragma once

#include "daemon/api/events.h"

#include <cstddef>

namespace daemon_control::handlers {

struct HandlerRegistryDependencies {
    daemon_core::api::EventDispatcher* dispatcher = nullptr;
};

class HandlerRegistry {
   public:
    explicit HandlerRegistry(HandlerRegistryDependencies deps);

    void registerDefaults();
    size_t registeredCount() const;

   private:
    HandlerRegistryDependencies deps_;
    size_t registered_{0};
};

}  // namespace daemon_control::handlers
