#include "audio/audio_io.h"

#include "logging/logger.h"

#include <cstring>
#include <iostream>

namespace AudioIO {

// WavReader implementation
WavReader::WavReader() : file_(nullptr) {
    std::memset(&info_, 0, sizeof(info_));
}

WavReader::~WavReader() {
    close();
}

bool WavReader::open(const std::string& filename) {
    file_ = sf_open(filename.c_str(), SFM_READ, &info_);
    if (!file_) {
        LOG_ERROR("Error opening input file: {}", filename);
        LOG_ERROR("libsndfile error: {}", sf_strerror(nullptr));
        return false;
    }

    std::cout << "Opened: " << filename << '\n';
    std::cout << "  Sample Rate: " << info_.samplerate << " Hz" << '\n';
    std::cout << "  Channels: " << info_.channels << '\n';
    std::cout << "  Frames: " << info_.frames << '\n';
    std::cout << "  Duration: " << static_cast<double>(info_.frames) / info_.samplerate
              << " seconds" << '\n';

    return true;
}

void WavReader::close() {
    if (file_) {
        sf_close(file_);
        file_ = nullptr;
    }
}

bool WavReader::readAll(AudioFile& output) {
    if (!file_) {
        LOG_ERROR("Error: File not opened");
        return false;
    }

    output.sampleRate = info_.samplerate;
    output.channels = info_.channels;
    output.frames = static_cast<int>(info_.frames);

    // Allocate buffer
    size_t totalSamples = info_.frames * info_.channels;
    output.data.resize(totalSamples);

    // Read all frames
    sf_count_t framesRead = sf_readf_float(file_, output.data.data(), info_.frames);
    if (framesRead != info_.frames) {
        LOG_ERROR("Error: Incomplete read. Expected {} frames, read {}", info_.frames, framesRead);
        LOG_ERROR("File may be corrupted or truncated.");
        return false;
    }

    return true;
}

bool WavReader::readBlock(float* buffer, sf_count_t frames) {
    if (!file_) {
        LOG_ERROR("Error: File not opened");
        return false;
    }

    sf_count_t framesRead = sf_readf_float(file_, buffer, frames);
    return framesRead == frames;
}

// WavWriter implementation
WavWriter::WavWriter() : file_(nullptr) {
    std::memset(&info_, 0, sizeof(info_));
}

WavWriter::~WavWriter() {
    close();
}

bool WavWriter::open(const std::string& filename, int sampleRate, int channels) {
    info_.samplerate = sampleRate;
    info_.channels = channels;
    info_.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    file_ = sf_open(filename.c_str(), SFM_WRITE, &info_);
    if (!file_) {
        LOG_ERROR("Error opening output file: {}", filename);
        LOG_ERROR("libsndfile error: {}", sf_strerror(nullptr));
        return false;
    }

    std::cout << "Created output file: " << filename << '\n';
    std::cout << "  Sample Rate: " << sampleRate << " Hz" << '\n';
    std::cout << "  Channels: " << channels << '\n';

    return true;
}

void WavWriter::close() {
    if (file_) {
        sf_close(file_);
        file_ = nullptr;
    }
}

bool WavWriter::writeAll(const AudioFile& input) {
    if (!file_) {
        LOG_ERROR("Error: File not opened");
        return false;
    }

    sf_count_t framesWritten = sf_writef_float(file_, input.data.data(), input.frames);
    if (framesWritten != input.frames) {
        LOG_ERROR("Error: Expected to write {} frames, wrote {}", input.frames, framesWritten);
        return false;
    }

    return true;
}

bool WavWriter::writeBlock(const float* buffer, sf_count_t frames) {
    if (!file_) {
        LOG_ERROR("Error: File not opened");
        return false;
    }

    sf_count_t framesWritten = sf_writef_float(file_, buffer, frames);
    return framesWritten == frames;
}

// Utility functions
namespace Utils {

void interleavedToSeparate(const float* interleaved, float* left, float* right, size_t frames) {
    for (size_t i = 0; i < frames; ++i) {
        left[i] = interleaved[i * 2];
        right[i] = interleaved[i * 2 + 1];
    }
}

void separateToInterleaved(const float* left, const float* right, float* interleaved,
                           size_t frames) {
    for (size_t i = 0; i < frames; ++i) {
        interleaved[i * 2] = left[i];
        interleaved[i * 2 + 1] = right[i];
    }
}

void monoToStereo(const float* mono, float* stereo, size_t frames) {
    for (size_t i = 0; i < frames; ++i) {
        stereo[i * 2] = mono[i];
        stereo[i * 2 + 1] = mono[i];
    }
}

void stereoToMono(const float* stereo, float* mono, size_t frames) {
    for (size_t i = 0; i < frames; ++i) {
        mono[i] = (stereo[i * 2] + stereo[i * 2 + 1]) * 0.5f;
    }
}

}  // namespace Utils

}  // namespace AudioIO
