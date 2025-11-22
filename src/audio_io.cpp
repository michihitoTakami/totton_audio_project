#include "audio_io.h"

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
        std::cerr << "Error opening input file: " << filename << std::endl;
        std::cerr << "libsndfile error: " << sf_strerror(nullptr) << std::endl;
        return false;
    }

    std::cout << "Opened: " << filename << std::endl;
    std::cout << "  Sample Rate: " << info_.samplerate << " Hz" << std::endl;
    std::cout << "  Channels: " << info_.channels << std::endl;
    std::cout << "  Frames: " << info_.frames << std::endl;
    std::cout << "  Duration: " << static_cast<double>(info_.frames) / info_.samplerate
              << " seconds" << std::endl;

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
        std::cerr << "Error: File not opened" << std::endl;
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
        std::cerr << "Error: Incomplete read. Expected " << info_.frames << " frames, read "
                  << framesRead << std::endl;
        std::cerr << "File may be corrupted or truncated." << std::endl;
        return false;
    }

    return true;
}

bool WavReader::readBlock(float* buffer, sf_count_t frames) {
    if (!file_) {
        std::cerr << "Error: File not opened" << std::endl;
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
        std::cerr << "Error opening output file: " << filename << std::endl;
        std::cerr << "libsndfile error: " << sf_strerror(nullptr) << std::endl;
        return false;
    }

    std::cout << "Created output file: " << filename << std::endl;
    std::cout << "  Sample Rate: " << sampleRate << " Hz" << std::endl;
    std::cout << "  Channels: " << channels << std::endl;

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
        std::cerr << "Error: File not opened" << std::endl;
        return false;
    }

    sf_count_t framesWritten = sf_writef_float(file_, input.data.data(), input.frames);
    if (framesWritten != input.frames) {
        std::cerr << "Error: Expected to write " << input.frames << " frames, wrote "
                  << framesWritten << std::endl;
        return false;
    }

    return true;
}

bool WavWriter::writeBlock(const float* buffer, sf_count_t frames) {
    if (!file_) {
        std::cerr << "Error: File not opened" << std::endl;
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
