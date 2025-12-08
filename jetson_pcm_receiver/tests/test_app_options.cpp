#include "app_options.h"

#include <cstdlib>
#include <gtest/gtest.h>
#include <optional>
#include <string>

namespace {

// 環境変数をテスト中だけ上書きし、スコープ終了時に元へ戻す
class ScopedEnv {
   public:
    ScopedEnv(const std::string &key, const std::string &value) : key_(key) {
        const char *prev = std::getenv(key.c_str());
        if (prev) {
            previous_ = prev;
        }
        setenv(key.c_str(), value.c_str(), 1);
    }

    ScopedEnv(const std::string &key) : key_(key) {
        const char *prev = std::getenv(key.c_str());
        if (prev) {
            previous_ = prev;
        }
        unsetenv(key.c_str());
    }

    ~ScopedEnv() {
        if (previous_) {
            setenv(key_.c_str(), previous_->c_str(), 1);
        } else {
            unsetenv(key_.c_str());
        }
    }

   private:
    std::string key_;
    std::optional<std::string> previous_;
};

}  // namespace

TEST(AppOptionsEnv, AppliesPortDeviceAndLogLevel) {
    ScopedEnv port("JPR_PORT", "50001");
    ScopedEnv device("JPR_DEVICE", "alsa:hw:USB,0,0");
    ScopedEnv logLevel("JPR_LOG_LEVEL", "debug");
    ScopedEnv ringFrames("JPR_RING_FRAMES", "4096");
    ScopedEnv watermark("JPR_WATERMARK_FRAMES", "1024");
    ScopedEnv zmqEndpoint("JPR_ZMQ_ENDPOINT", "ipc:///tmp/test.sock");
    ScopedEnv zmqPub("JPR_ZMQ_PUB_INTERVAL_MS", "250");

    AppOptions options = makeDefaultOptions();
    std::string error;
    ASSERT_TRUE(applyEnvOverrides(options, error)) << error;

    EXPECT_EQ(options.port, 50001);
    EXPECT_EQ(options.device.alsaName, "hw:USB,0,0");
    EXPECT_EQ(options.logLevel, LogLevel::Debug);
    EXPECT_EQ(options.ringBufferFrames, 4096u);
    EXPECT_EQ(options.watermarkFrames, 1024u);
    EXPECT_EQ(options.zmqEndpoint, "ipc:///tmp/test.sock");
    EXPECT_EQ(options.zmqPublishIntervalMs, 250);
}

TEST(AppOptionsEnv, RejectsInvalidDevice) {
    ScopedEnv device("JPR_DEVICE", "alsa:");

    AppOptions options = makeDefaultOptions();
    std::string error;
    EXPECT_FALSE(applyEnvOverrides(options, error));
    EXPECT_FALSE(error.empty());
}

TEST(AppOptionsEnv, DisablesZmqAndParsesPriorityClients) {
    ScopedEnv disableZmq("JPR_DISABLE_ZMQ", "true");
    ScopedEnv priority("JPR_PRIORITY_CLIENTS", "10.0.0.1,192.168.1.10");
    ScopedEnv connection("JPR_CONNECTION_MODE", "priority");

    AppOptions options = makeDefaultOptions();
    std::string error;
    ASSERT_TRUE(applyEnvOverrides(options, error)) << error;

    EXPECT_FALSE(options.enableZmq);
    ASSERT_EQ(options.priorityClients.size(), 2u);
    EXPECT_EQ(options.priorityClients[0], "10.0.0.1");
    EXPECT_EQ(options.priorityClients[1], "192.168.1.10");
    EXPECT_EQ(options.connectionMode, ConnectionMode::Priority);
}
