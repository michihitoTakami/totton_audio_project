#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct FilterMetadata {
    std::size_t taps = 0;
    uint32_t inputRate = 0;
    uint32_t outputRate = 0;
    uint32_t upsampleRatio = 0;
};

struct VulkanOverlapSaveOptions {
    std::string inputPath;
    std::string outputPath;
    std::string filterPath;
    std::string filterMetadataPath;
    uint32_t fftSizeOverride = 0;
    uint32_t chunkFrames = 8192;
};

// メタデータ(JSON)と係数(bin)を読み込むヘルパ
bool loadFilterMetadata(const std::string& jsonPath, FilterMetadata& out);
bool loadFilterCoefficients(const std::string& binPath, std::size_t taps, std::vector<float>& out);

// バッファ入力（モノラル）をVulkan Overlap-Saveでアップサンプル+FIR畳み込みする
// output にはアップサンプル後の波形が追加される
bool processOverlapSaveBuffer(const std::vector<float>& inputMono,
                              const std::vector<float>& filterTaps, uint32_t upsampleRatio,
                              uint32_t fftSize, uint32_t chunkFrames, std::vector<float>& output);

// ステレオ入力（インターリーブ）をVulkan Overlap-Saveで処理する
bool processOverlapSaveStereoBuffer(const std::vector<float>& inputInterleaved,
                                    const std::vector<float>& filterTaps, uint32_t upsampleRatio,
                                    uint32_t fftSize, uint32_t chunkFrames,
                                    std::vector<float>& outputInterleaved);

// WAV入出力付きの高レベルCLI処理
int runVulkanOverlapSave(const VulkanOverlapSaveOptions& opts);
