#include "daemon/api/dependencies.h"
#include "daemon/api/events.h"
#include "daemon/audio_pipeline/filter_manager.h"
#include "daemon/audio_pipeline/rate_switcher.h"
#include "daemon/audio_pipeline/soft_mute_runner.h"
#include "daemon/control/handlers/handler_registry.h"
#include "daemon/input/pipewire_input.h"
#include "daemon/input/rtp_input_adapter.h"
#include "daemon/output/alsa_output.h"

#include <gtest/gtest.h>

#include <atomic>
#include <string>

namespace {

daemon::api::RateChangeRequested makeRateEvent(int rate) {
    daemon::api::RateChangeRequested event;
    event.detectedInputRate = rate;
    event.rateFamily = ConvolutionEngine::RateFamily::RATE_48K;
    return event;
}

daemon::api::FilterSwitchRequested makeFilterEvent(const std::string& path) {
    daemon::api::FilterSwitchRequested event;
    event.filterPath = path;
    event.phaseType = PhaseType::Linear;
    event.reloadHeadroom = true;
    return event;
}

daemon::api::DeviceChangeRequested makeDeviceEvent(const std::string& device) {
    daemon::api::DeviceChangeRequested event;
    event.preferredDevice = device;
    event.mode = OutputMode::Usb;
    return event;
}

}  // namespace

TEST(DaemonSkeleton, RateChangeEventUpdatesState) {
    daemon::api::EventDispatcher dispatcher;
    std::atomic<int> pending{0};
    std::atomic<int> currentInput{0};
    std::atomic<int> currentOutput{0};

    audio_pipeline::RateSwitcher switcher(
        {.dispatcher = &dispatcher,
         .deps = {.currentInputRate = &currentInput, .currentOutputRate = &currentOutput},
         .pendingRate = &pending});
    switcher.start();

    dispatcher.publish(makeRateEvent(48000));

    EXPECT_EQ(switcher.lastSeenRate(), 48000);
    EXPECT_EQ(pending.load(), 48000);
    EXPECT_EQ(currentInput.load(), 48000);
    EXPECT_EQ(currentOutput.load(), 48000);
}

TEST(DaemonSkeleton, FilterSwitchRoutesToManagers) {
    daemon::api::EventDispatcher dispatcher;
    bool refreshed = false;
    audio_pipeline::FilterManager manager(
        {.dispatcher = &dispatcher,
         .deps = {.refreshHeadroom = [&](const std::string&) { refreshed = true; }}});
    audio_pipeline::SoftMuteRunner runner({.dispatcher = &dispatcher});

    manager.start();
    runner.start();

    dispatcher.publish(makeFilterEvent("data/coefficients/filter.bin"));

    EXPECT_EQ(manager.lastRequestedPath(), "data/coefficients/filter.bin");
    EXPECT_EQ(manager.lastRequestedPhase(), PhaseType::Linear);
    EXPECT_TRUE(refreshed);
    EXPECT_TRUE(runner.wasTriggered());
}

TEST(DaemonSkeleton, DeviceChangeRoutesToOutput) {
    daemon::api::EventDispatcher dispatcher;
    std::atomic<bool> outputReady{true};

    daemon_output::AlsaOutput output(
        {.dispatcher = &dispatcher, .deps = {.outputReady = &outputReady}});
    output.start();

    dispatcher.publish(makeDeviceEvent("hw:TestDAC"));

    EXPECT_EQ(output.lastRequestedDevice(), "hw:TestDAC");
    EXPECT_FALSE(outputReady.load());
}

TEST(DaemonSkeleton, ControlHandlersRegisterSubscriptions) {
    daemon::api::EventDispatcher dispatcher;
    daemon_control::handlers::HandlerRegistry registry({.dispatcher = &dispatcher});
    registry.registerDefaults();

    daemon::api::RateChangeRequested rateEvent = makeRateEvent(96000);
    dispatcher.publish(rateEvent);

    EXPECT_EQ(registry.registeredCount(), 3u);
}

TEST(DaemonSkeleton, RtpAdapterPublishesDeviceChange) {
    daemon::api::EventDispatcher dispatcher;
    std::string lastDevice;
    dispatcher.subscribe([&](const daemon::api::DeviceChangeRequested& event) {
        lastDevice = event.preferredDevice;
    });

    daemon_input::RtpInputAdapter adapter({.dispatcher = &dispatcher});
    adapter.requestDeviceChange("hw:RTP");

    EXPECT_EQ(lastDevice, "hw:RTP");
    EXPECT_EQ(adapter.lastRequestedDevice(), "hw:RTP");
}

TEST(DaemonSkeleton, PipeWireInputPublishesRateChange) {
    daemon::api::EventDispatcher dispatcher;
    std::atomic<bool> running{false};
    std::atomic<int> pending{0};
    int observedRate = 0;

    dispatcher.subscribe([&](const daemon::api::RateChangeRequested& event) {
        observedRate = event.detectedInputRate;
    });

    daemon_input::PipeWireInput input({.dispatcher = &dispatcher,
                                       .runningFlag = &running,
                                       .pendingRate = &pending});
    input.start();
    input.publishRateChange(44100, ConvolutionEngine::RateFamily::RATE_44K);

    EXPECT_TRUE(input.isRunning());
    EXPECT_TRUE(running.load());
    EXPECT_EQ(pending.load(), 44100);
    EXPECT_EQ(observedRate, 44100);
}


