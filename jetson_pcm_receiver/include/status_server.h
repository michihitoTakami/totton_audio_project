#pragma once

#include <atomic>
#include <string>
#include <thread>

struct PcmStatusSnapshot {
    bool connected{false};
    std::string lastHeaderSummary;
    std::size_t bufferedFrames{0};
    std::size_t maxBufferedFrames{0};
    std::size_t droppedFrames{0};
    std::size_t xrunCount{0};
};

// 非常に簡易なHTTPステータスサーバ。ローカルバインドのみ。
class StatusServer {
   public:
    StatusServer(int port, std::atomic_bool &stopFlag);
    ~StatusServer();

    void start();
    void stop();

    // 最新状態をセット（呼び出し側で定期的に更新）
    void setSnapshot(const PcmStatusSnapshot &snapshot);

   private:
    int port_;
    int fd_{-1};
    std::atomic_bool &stopFlag_;
    std::atomic_bool running_{false};
    std::thread worker_;
    PcmStatusSnapshot snapshot_;

    void serveLoop();
};
