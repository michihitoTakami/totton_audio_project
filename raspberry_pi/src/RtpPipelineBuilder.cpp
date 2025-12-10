// Construct gst-launch command lines for RTP sender
#include "RtpPipelineBuilder.h"

#include <sstream>

std::vector<std::string> RtpPipelineBuilder::build(const RtpPipelineConfig &config,
                                                   const CaptureParams &params) {
    const std::string payloader = payloaderForFormat(params.format);
    const std::string encoding = encodingName(params.format);
    const std::string rawFmt = rawFormat(params.format);
    const std::string caps = "audio/x-raw,rate=" + std::to_string(params.sampleRate) +
                             ",channels=" + std::to_string(params.channels) + ",format=" + rawFmt;
    const std::string rtpCaps =
        "application/x-rtp,media=audio,clock-rate=" + std::to_string(params.sampleRate) +
        ",encoding-name=" + encoding + ",payload=" + std::to_string(config.payloadType) +
        ",channels=" + std::to_string(params.channels);

    return {"gst-launch-1.0",
            "-e",
            "rtpbin",
            "name=rtpbin",
            "ntp-sync=true",
            "buffer-mode=sync",
            "alsasrc",
            "device=" + config.device,
            "!",
            "audioresample",
            "quality=10",
            "!",
            "audioconvert",
            "!",
            caps,
            "!",
            payloader,
            "pt=" + std::to_string(config.payloadType),
            "!",
            rtpCaps,
            "!",
            "rtpbin.send_rtp_sink_0",
            "rtpbin.send_rtp_src_0",
            "!",
            "udpsink",
            "host=" + config.host,
            "port=" + std::to_string(config.rtpPort),
            "sync=true",
            "async=false",
            "rtpbin.send_rtcp_src_0",
            "!",
            "udpsink",
            "host=" + config.host,
            "port=" + std::to_string(config.rtcpSendPort),
            "sync=false",
            "async=false",
            "udpsrc",
            "port=" + std::to_string(config.rtcpListenPort),
            "!",
            "rtpbin.recv_rtcp_sink_0"};
}

std::string RtpPipelineBuilder::toCommandString(const std::vector<std::string> &args) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (i > 0) {
            oss << ' ';
        }
        oss << args[i];
    }
    return oss.str();
}

std::string RtpPipelineBuilder::payloaderForFormat(AlsaCapture::SampleFormat fmt) {
    switch (fmt) {
    case AlsaCapture::SampleFormat::S16_LE:
        return "rtpL16pay";
    case AlsaCapture::SampleFormat::S24_3LE:
        return "rtpL24pay";
    case AlsaCapture::SampleFormat::S32_LE:
        return "rtpL32pay";
    }
    return "rtpL24pay";
}

std::string RtpPipelineBuilder::encodingName(AlsaCapture::SampleFormat fmt) {
    switch (fmt) {
    case AlsaCapture::SampleFormat::S16_LE:
        return "L16";
    case AlsaCapture::SampleFormat::S24_3LE:
        return "L24";
    case AlsaCapture::SampleFormat::S32_LE:
        return "L32";
    }
    return "L24";
}

std::string RtpPipelineBuilder::rawFormat(AlsaCapture::SampleFormat fmt) {
    switch (fmt) {
    case AlsaCapture::SampleFormat::S16_LE:
        return "S16LE";
    case AlsaCapture::SampleFormat::S24_3LE:
        return "S24LE";
    case AlsaCapture::SampleFormat::S32_LE:
        return "S32LE";
    }
    return "S24LE";
}
