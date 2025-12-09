#include "daemon/shutdown_manager.h"

#include <chrono>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#ifdef HAVE_SYSTEMD
#include <systemd/sd-daemon.h>
#endif

namespace shutdown_manager {

ShutdownManager::ShutdownManager(Dependencies deps) : deps_(std::move(deps)) {
    if (!deps_.runningFlag || !deps_.reloadFlag) {
        throw std::invalid_argument("ShutdownManager requires running/reload flags");
    }

    controller_.setSignalState(&GracefulShutdown::getGlobalSignalState());
    controller_.setFadeOutCallback([this]() {
        if (auto* softMute = activeSoftMute()) {
            softMute->startFadeOut();
        }
    });
    controller_.setLogCallback([](const char* message) { std::cout << message << std::endl; });
    controller_.setQuitLoopCallback([this]() {
        if (deps_.mainLoopRunningFlag && deps_.mainLoopRunningFlag->load() && quitLoopCallback_) {
            quitLoopCallback_();
        }
    });
}

void ShutdownManager::installSignalHandlers() {
    std::signal(SIGINT, GracefulShutdown::signalHandler);
    std::signal(SIGTERM, GracefulShutdown::signalHandler);
    std::signal(SIGHUP, GracefulShutdown::signalHandler);
}

void ShutdownManager::setQuitLoopCallback(std::function<void()> cb) {
    quitLoopCallback_ = std::move(cb);
}

void ShutdownManager::setMainLoopRunning(bool running) {
    if (deps_.mainLoopRunningFlag) {
        deps_.mainLoopRunningFlag->store(running);
    }
    controller_.setMainLoopRunning(running);
}

void ShutdownManager::notifyReady() {
#ifdef HAVE_SYSTEMD
    if (!readyNotified_) {
        sendReadyNotify();
    }
#endif
}

void ShutdownManager::tick() {
    bool processed = controller_.processPendingSignals();
    if (processed) {
        deps_.runningFlag->store(controller_.isRunning());
        deps_.reloadFlag->store(controller_.isReloadRequested());
    }
#ifdef HAVE_SYSTEMD
    if (readyNotified_ && controller_.isRunning()) {
        sendWatchdog();
    }
#endif
}

void ShutdownManager::runShutdownSequence() {
    if (sequenceRan_) {
        return;
    }
    sequenceRan_ = true;

    std::cout << "Shutting down..." << std::endl;

#ifdef HAVE_SYSTEMD
    sendStoppingNotify();
#endif

    waitForFadeOut();
}

void ShutdownManager::reset() {
    sequenceRan_ = false;
    readyNotified_ = false;
    stoppingNotified_ = false;
    deps_.runningFlag->store(true);
    deps_.reloadFlag->store(false);
    if (deps_.mainLoopRunningFlag) {
        deps_.mainLoopRunningFlag->store(false);
    }
    controller_.setRunning(true);
    controller_.clearReloadRequest();
    GracefulShutdown::getGlobalSignalState().reset();
}

bool ShutdownManager::isRunning() const {
    return controller_.isRunning();
}

bool ShutdownManager::isReloadRequested() const {
    return controller_.isReloadRequested();
}

SoftMute::Controller* ShutdownManager::activeSoftMute() const {
    if (!deps_.softMute) {
        return nullptr;
    }
    return *deps_.softMute;
}

void ShutdownManager::waitForFadeOut() {
    auto* softMute = activeSoftMute();
    if (!softMute) {
        return;
    }

    if (!softMute->isTransitioning()) {
        return;
    }

    std::cout << "  Step 2: Waiting for fade-out to complete..." << std::endl;
    auto fadeStart = std::chrono::steady_clock::now();
    while (softMute->isTransitioning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (std::chrono::steady_clock::now() - fadeStart > std::chrono::milliseconds(100)) {
            std::cout << "  Step 2: Fade-out timeout, forcing shutdown" << std::endl;
            break;
        }
    }
}

void ShutdownManager::sendWatchdog() {
#ifdef HAVE_SYSTEMD
    sd_notify(0, "WATCHDOG=1");
#endif
}

#ifdef HAVE_SYSTEMD
void ShutdownManager::sendReadyNotify() {
    sd_notify(0, "READY=1\nSTATUS=Processing audio...\n");
    readyNotified_ = true;
    std::cout << "systemd: Notified READY=1" << std::endl;
}

void ShutdownManager::sendStoppingNotify() {
    if (!stoppingNotified_) {
        sd_notify(0, "STOPPING=1\nSTATUS=Shutting down...\n");
        stoppingNotified_ = true;
        std::cout << "systemd: Notified STOPPING=1" << std::endl;
    }
}
#endif

}  // namespace shutdown_manager
