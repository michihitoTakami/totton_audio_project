#pragma once

#include <string>

// クライアント接続制御のモード。
enum class ConnectionMode {
    Single,    // 従来どおり同時1接続のみ（追加接続は拒否）
    Takeover,  // 新規接続で既存をキックアウト
    Priority,  // 優先クライアントのみキックアウト可能
};

inline std::string toString(ConnectionMode mode) {
    switch (mode) {
    case ConnectionMode::Single:
        return "single";
    case ConnectionMode::Takeover:
        return "takeover";
    case ConnectionMode::Priority:
        return "priority";
    }
    return "unknown";
}

inline ConnectionMode parseConnectionMode(const std::string& value) {
    if (value == "single") {
        return ConnectionMode::Single;
    }
    if (value == "takeover" || value == "replace") {
        return ConnectionMode::Takeover;
    }
    if (value == "priority" || value == "priority-only") {
        return ConnectionMode::Priority;
    }
    return ConnectionMode::Single;
}
