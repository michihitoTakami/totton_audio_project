// Build gst-launch-1.0 command for RTP send pipeline
#pragma once

#include "AlsaCapture.h"
#include "HwParamsMonitor.h"
#include "RtpOptions.h"

#include <string>
#include <vector>

struct RtpPipelineConfig {
    std::string host;
    std::uint16_t rtpPort{46000};
    std::uint16_t rtcpSendPort{46001};
    std::uint16_t rtcpListenPort{46002};
    int payloadType{96};
    std::string device;
    int latencyMs{100};
};

class RtpPipelineBuilder {
   public:
    static std::vector<std::string> build(const RtpPipelineConfig &config,
                                          const CaptureParams &params);
    static std::string toCommandString(const std::vector<std::string> &args);

   private:
    static std::string payloaderForFormat(AlsaCapture::SampleFormat fmt);
    static std::string encodingName(AlsaCapture::SampleFormat fmt);
    static std::string rawFormat(AlsaCapture::SampleFormat fmt);
};
