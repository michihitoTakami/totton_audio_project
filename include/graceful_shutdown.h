#pragma once

#include <atomic>
#include <csignal>
#include <functional>

namespace GracefulShutdown {

// ========== Signal State ==========
// These are the signal flags that are set by the signal handler
// and polled by the main loop. Using volatile sig_atomic_t ensures
// async-signal-safe access.

struct SignalState {
    volatile sig_atomic_t shutdown = 0;  // SIGTERM, SIGINT
    volatile sig_atomic_t reload = 0;    // SIGHUP
    volatile sig_atomic_t received = 0;  // Last signal number (for logging)

    // Reset all flags
    void reset() {
        shutdown = 0;
        reload = 0;
        received = 0;
    }
};

// ========== Shutdown Controller ==========
// Manages the graceful shutdown state and signal processing.
// This class is designed to be testable without actual signal delivery.

class Controller {
   public:
    // Callbacks for shutdown actions
    using FadeOutCallback = std::function<void()>;
    using QuitLoopCallback = std::function<void()>;
    using LogCallback = std::function<void(const char*)>;

    Controller() = default;

    // Set the signal state (for testing, this can be a mock)
    void setSignalState(SignalState* state) {
        signalState_ = state;
    }

    // Set callbacks for shutdown actions
    void setFadeOutCallback(FadeOutCallback cb) {
        fadeOutCallback_ = std::move(cb);
    }
    void setQuitLoopCallback(QuitLoopCallback cb) {
        quitLoopCallback_ = std::move(cb);
    }
    void setLogCallback(LogCallback cb) {
        logCallback_ = std::move(cb);
    }

    // Process pending signals. Returns true if any signal was processed.
    // IMPORTANT: Shutdown (SIGTERM/SIGINT) takes priority over reload (SIGHUP).
    bool processPendingSignals();

    // Check if reload was requested (for do-while loop)
    bool isReloadRequested() const {
        return reloadRequested_.load();
    }

    // Check if running
    bool isRunning() const {
        return running_.load();
    }

    // Set running state
    void setRunning(bool running) {
        running_ = running;
    }

    // Set main loop running state
    void setMainLoopRunning(bool running) {
        mainLoopRunning_ = running;
    }
    bool isMainLoopRunning() const {
        return mainLoopRunning_.load();
    }

    // Clear reload request (used when shutdown takes priority)
    void clearReloadRequest() {
        reloadRequested_ = false;
    }

    // Get last processed signal number (for logging)
    int getLastSignal() const {
        return lastSignal_;
    }

    // Get action taken in last processPendingSignals call
    enum class Action { NONE, SHUTDOWN, RELOAD };
    Action getLastAction() const {
        return lastAction_;
    }

   private:
    SignalState* signalState_ = nullptr;
    std::atomic<bool> running_{true};
    std::atomic<bool> reloadRequested_{false};
    std::atomic<bool> mainLoopRunning_{false};

    FadeOutCallback fadeOutCallback_;
    QuitLoopCallback quitLoopCallback_;
    LogCallback logCallback_;

    int lastSignal_ = 0;
    Action lastAction_ = Action::NONE;
};

// ========== Signal Handler ==========
// Async-signal-safe signal handler that only sets flags.
// Must be installed with std::signal() or sigaction().
void signalHandler(int sig);

// Get the global signal state (for use with signalHandler)
SignalState& getGlobalSignalState();

}  // namespace GracefulShutdown
