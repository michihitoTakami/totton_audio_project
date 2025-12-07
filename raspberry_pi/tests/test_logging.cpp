#include "logging.h"

#include <gtest/gtest.h>

TEST(Logging, ParsesLevelNames) {
    EXPECT_EQ(parseLogLevel("info"), LogLevel::Info);
    EXPECT_EQ(parseLogLevel("INFO"), LogLevel::Info);
    EXPECT_EQ(parseLogLevel("warn"), LogLevel::Warn);
    EXPECT_EQ(parseLogLevel("ERROR"), LogLevel::Error);
    EXPECT_EQ(parseLogLevel("debug"), LogLevel::Debug);
    EXPECT_EQ(parseLogLevel("unknown"), LogLevel::Info);
}

TEST(Logging, HonorsConfiguredLevel) {
    setLogLevel(LogLevel::Warn);
    EXPECT_TRUE(shouldLog(LogLevel::Error));
    EXPECT_TRUE(shouldLog(LogLevel::Warn));
    EXPECT_FALSE(shouldLog(LogLevel::Info));

    setLogLevel(LogLevel::Debug);
    EXPECT_TRUE(shouldLog(LogLevel::Info));

    setLogLevel(LogLevel::Info);  // reset for other tests
}
