#pragma once

#include "connection_mode.h"
#include "logging.h"
#include "output_device.h"

#include <cstddef>
#include <string>
#include <vector>

struct AppOptions {
    int port = 46001;
    OutputDeviceSpec device{};
    LogLevel logLevel = LogLevel::Info;
    std::size_t ringBufferFrames = 8192;  // 0で無効
    std::size_t watermarkFrames = 0;      // 0で自動 (75%)
    bool enableZmq = true;
    bool enableZmqRateNotify = true;
    std::string zmqEndpoint = "ipc:///tmp/jetson_pcm_receiver.sock";
    std::string zmqToken;
    int zmqPublishIntervalMs = 1000;
    int recvTimeoutMs = 250;
    int recvTimeoutSleepMs = 50;
    int acceptCooldownMs = 250;
    int maxConsecutiveTimeouts = 3;
    ConnectionMode connectionMode = ConnectionMode::Single;
    std::vector<std::string> priorityClients;
};

// デフォルトオプション（ループバックを安全に初期化）
AppOptions makeDefaultOptions();

// 環境変数の値でオプションを上書きする。失敗時はfalseとエラーメッセージを返す。
bool applyEnvOverrides(AppOptions &options, std::string &error);

// CLI引数を解析してオプションへ反映する。showHelp=trueの場合はヘルプ表示のみ。
bool parseArgs(int argc, char **argv, AppOptions &options, bool &showHelp, std::string &error);

// ヘルプ表示
void printHelp(const char *exeName);
