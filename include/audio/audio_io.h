#ifndef AUDIO_IO_H
#define AUDIO_IO_H

#include <memory>
#include <sndfile.h>
#include <string>
#include <vector>

namespace AudioIO {

struct AudioFile {
    std::vector<float> data;  // Interleaved audio data
    int sampleRate;
    int channels;
    int frames;

    AudioFile() : sampleRate(0), channels(0), frames(0) {}
};

class WavReader {
   public:
    WavReader();
    ~WavReader();

    bool open(const std::string& filename);
    void close();

    int getSampleRate() const {
        return info_.samplerate;
    }
    int getChannels() const {
        return info_.channels;
    }
    sf_count_t getFrames() const {
        return info_.frames;
    }

    // Read all audio data at once
    bool readAll(AudioFile& output);

    // Read audio data in blocks
    bool readBlock(float* buffer, sf_count_t frames);

   private:
    SNDFILE* file_;
    SF_INFO info_;
};

class WavWriter {
   public:
    WavWriter();
    ~WavWriter();

    bool open(const std::string& filename, int sampleRate, int channels);
    void close();

    // Write all audio data at once
    bool writeAll(const AudioFile& input);

    // Write audio data in blocks
    bool writeBlock(const float* buffer, sf_count_t frames);

   private:
    SNDFILE* file_;
    SF_INFO info_;
};

// Utility functions
namespace Utils {
// Convert stereo interleaved to separate L/R channels
void interleavedToSeparate(const float* interleaved, float* left, float* right, size_t frames);

// Convert separate L/R channels to stereo interleaved
void separateToInterleaved(const float* left, const float* right, float* interleaved,
                           size_t frames);

// Convert mono to stereo (duplicate channel)
void monoToStereo(const float* mono, float* stereo, size_t frames);

// Mix stereo to mono (average)
void stereoToMono(const float* stereo, float* mono, size_t frames);
}  // namespace Utils

}  // namespace AudioIO

#endif  // AUDIO_IO_H
