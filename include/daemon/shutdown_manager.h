#pragma once

#include "graceful_shutdown.h"
#include "soft_mute.h"

#include <atomic>
#include <functional>

namespace shutdown_manager {

class ShutdownManager {
   public:
    struct Dependencies {
        SoftMute::Controller** softMute = nullptr;
        std::atomic<bool>* runningFlag = nullptr;
        std::atomic<bool>* reloadFlag = nullptr;
        std::atomic<bool>* mainLoopRunningFlag = nullptr;
    };

    explicit ShutdownManager(Dependencies deps);

    // Signal handling
    void installSignalHandlers();

    // Shutdown callbacks
    void setQuitLoopCallback(std::function<void()> cb);

    void setMainLoopRunning(bool running);

    // Notifications
    void notifyReady();

    // Periodic processing (called from main loops)
    void tick();

    // Shutdown execution
    void runShutdownSequence();

    // Reset manager state between restarts
    void reset();

    bool isRunning() const;
    bool isReloadRequested() const;

   private:
    SoftMute::Controller* activeSoftMute() const;
    void waitForFadeOut();
    void sendWatchdog();
#ifdef HAVE_SYSTEMD
    void sendReadyNotify();
    void sendStoppingNotify();
#endif

    Dependencies deps_;
    GracefulShutdown::Controller controller_;  // own signal controller
    std::function<void()> quitLoopCallback_;

    bool readyNotified_{false};
    bool stoppingNotified_{false};
    bool sequenceRan_{false};
};

}  // namespace shutdown_manager
