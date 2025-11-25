#include "graceful_shutdown.h"

#include <cstdio>

namespace GracefulShutdown {

// Global signal state (used by signalHandler)
static SignalState g_signalState;

SignalState& getGlobalSignalState() {
    return g_signalState;
}

// Async-signal-safe signal handler - ONLY sets flags
void signalHandler(int sig) {
    g_signalState.received = sig;
    if (sig == SIGHUP) {
        g_signalState.reload = 1;
    } else {
        g_signalState.shutdown = 1;
    }
}

bool Controller::processPendingSignals() {
    if (!signalState_) {
        return false;
    }

    lastAction_ = Action::NONE;

    // Check for shutdown request FIRST (SIGTERM, SIGINT) - takes priority over reload
    if (signalState_->shutdown) {
        signalState_->shutdown = 0;
        signalState_->reload = 0;  // Clear any pending reload - shutdown takes priority
        lastSignal_ = signalState_->received;
        lastAction_ = Action::SHUTDOWN;

        // Log via callback (avoids std::cout blocking in realtime context)
        if (logCallback_) {
            char buf[64];
            snprintf(buf, sizeof(buf), "\nReceived signal %d, shutting down...", lastSignal_);
            logCallback_(buf);
        }

        // Clear reload flag to ensure clean shutdown (not restart)
        // This prevents do { ... } while (reloadRequested) from restarting
        reloadRequested_ = false;

        // Start fade-out for glitch-free shutdown
        if (fadeOutCallback_) {
            fadeOutCallback_();
        }

        // Quit main loop to trigger shutdown sequence
        if (mainLoopRunning_.load() && quitLoopCallback_) {
            quitLoopCallback_();
        }
        // Always set running_ = false as fallback (even if callback called)
        // This ensures we stop even if quitLoopCallback_ fails or is no-op
        running_ = false;
        return true;
    }

    // Check for reload request (SIGHUP) - only if no shutdown pending
    if (signalState_->reload) {
        signalState_->reload = 0;
        lastSignal_ = signalState_->received;
        lastAction_ = Action::RELOAD;

        // Log via callback (avoids std::cout blocking in realtime context)
        if (logCallback_) {
            char buf[80];
            snprintf(buf, sizeof(buf),
                     "\nReceived SIGHUP (signal %d), restarting for config reload...", lastSignal_);
            logCallback_(buf);
        }
        reloadRequested_ = true;

        // Start fade-out for glitch-free reload
        if (fadeOutCallback_) {
            fadeOutCallback_();
        }

        // Quit main loop to trigger reload sequence
        if (mainLoopRunning_.load() && quitLoopCallback_) {
            quitLoopCallback_();
        }
        // Always set running_ = false as fallback (even if callback called)
        // This ensures we stop even if quitLoopCallback_ fails or is no-op
        running_ = false;
        return true;
    }

    return false;
}

}  // namespace GracefulShutdown
