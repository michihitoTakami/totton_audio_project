#include "delimiter/inference_backend.h"

#include "logging/logger.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <utility>
#include <vector>

#ifdef DELIMITER_ENABLE_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace delimiter {
namespace {

constexpr uint32_t kDelimiterRate44k = 44100;
constexpr uint32_t kDelimiterRate48k = 48000;

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

[[maybe_unused]] bool isSupportedDelimiterRate(uint32_t rate) {
    return rate == kDelimiterRate44k || rate == kDelimiterRate48k;
}

class BypassInferenceBackend final : public InferenceBackend {
   public:
    explicit BypassInferenceBackend(uint32_t expectedSampleRate)
        : expectedSampleRate_(expectedSampleRate) {}

    const char* name() const override {
        return "bypass";
    }

    uint32_t expectedSampleRate() const override {
        return expectedSampleRate_;
    }

    InferenceResult process(const StereoPlanarView& input, std::vector<float>& outLeft,
                            std::vector<float>& outRight) override {
        if (!input.valid() || input.frames == 0) {
            outLeft.clear();
            outRight.clear();
            return {InferenceStatus::InvalidConfig, "invalid input buffer"};
        }

        outLeft.assign(input.left, input.left + input.frames);
        outRight.assign(input.right, input.right + input.frames);
        return {InferenceStatus::Ok, ""};
    }

    void reset() override {}

   private:
    uint32_t expectedSampleRate_;
};

#ifdef DELIMITER_ENABLE_ORT

Ort::Env& ortEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "delimiter");
    return env;
}

std::string statusMessage(OrtStatus* status, const std::string& prefix) {
    std::string message = prefix;
    message += Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    return message;
}

class OrtInferenceBackend final : public InferenceBackend {
   public:
    explicit OrtInferenceBackend(AppConfig::DelimiterConfig config)
        : config_(std::move(config)),
          expectedSampleRate_(config_.expectedSampleRate),
          memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
        initialize();
    }

    const char* name() const override {
        return "ort";
    }

    uint32_t expectedSampleRate() const override {
        return expectedSampleRate_;
    }

    InferenceResult process(const StereoPlanarView& input, std::vector<float>& outLeft,
                            std::vector<float>& outRight) override {
        outLeft.clear();
        outRight.clear();

        if (!input.valid() || input.frames == 0) {
            return {InferenceStatus::InvalidConfig, "invalid input buffer"};
        }
        if (!session_) {
            return {InferenceStatus::InvalidConfig,
                    initError_.empty() ? "ORT session is not initialized" : initError_};
        }

        inputBuffer_.resize(input.frames * 2);
        for (std::size_t i = 0; i < input.frames; ++i) {
            inputBuffer_[i * 2] = input.left[i];
            inputBuffer_[i * 2 + 1] = input.right[i];
        }

        std::array<int64_t, 3> shape{1, 2, static_cast<int64_t>(input.frames)};
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo_, inputBuffer_.data(), inputBuffer_.size(), shape.data(), shape.size());

        const char* inputNames[] = {inputName_.c_str()};

        try {
            auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1,
                                         outputNamePtrs_.data(), outputNamePtrs_.size());
            if (outputs.empty()) {
                return {InferenceStatus::Error, "onnxruntime returned no outputs"};
            }
            return extractOutputs(outputs.back(), outLeft, outRight);
        } catch (const Ort::Exception& e) {
            return {InferenceStatus::Error, e.what()};
        } catch (const std::exception& e) {
            return {InferenceStatus::Error, e.what()};
        }
    }

    void reset() override {
        inputBuffer_.clear();
    }

   private:
    enum class ProviderType { Cpu, Cuda, Tensorrt };

    struct Provider {
        ProviderType type = ProviderType::Cpu;
        std::string ortName;
        bool valid = false;
    };

    void initialize() {
        if (!isSupportedDelimiterRate(expectedSampleRate_)) {
            initError_ = "delimiter.expectedSampleRate must be 44100 or 48000";
            return;
        }
        if (config_.ort.modelPath.empty()) {
            initError_ = "delimiter.ort.modelPath is empty";
            return;
        }
        if (!std::filesystem::exists(config_.ort.modelPath)) {
            initError_ = "delimiter.ort.modelPath does not exist: " + config_.ort.modelPath;
            return;
        }

        Provider provider = parseProvider(config_.ort.provider);
        if (!provider.valid) {
            initError_ = "Unsupported ORT provider: " + config_.ort.provider;
            return;
        }

        if (!isProviderAvailable(provider.ortName)) {
            initError_ = "Execution provider '" + provider.ortName +
                         "' is not available in this onnxruntime build";
            return;
        }

        try {
            Ort::SessionOptions options;
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            if (config_.ort.intraOpThreads > 0) {
                options.SetIntraOpNumThreads(config_.ort.intraOpThreads);
            }

            if (!appendProvider(options, provider)) {
                return;
            }

            session_ =
                std::make_unique<Ort::Session>(ortEnv(), config_.ort.modelPath.c_str(), options);
            loadIoNames();
        } catch (const Ort::Exception& e) {
            initError_ = e.what();
            session_.reset();
        } catch (const std::exception& e) {
            initError_ = e.what();
            session_.reset();
        }
    }

    Provider parseProvider(const std::string& providerStr) const {
        std::string lower = toLower(providerStr);
        if (lower == "cpu") {
            return Provider{ProviderType::Cpu, "CPUExecutionProvider", true};
        }
        if (lower == "cuda") {
            return Provider{ProviderType::Cuda, "CUDAExecutionProvider", true};
        }
        if (lower == "tensorrt" || lower == "trt") {
            return Provider{ProviderType::Tensorrt, "TensorrtExecutionProvider", true};
        }
        return Provider{};
    }

    bool isProviderAvailable(const std::string& providerName) const {
        try {
            std::vector<std::string> providers = Ort::GetAvailableProviders();
            return std::find(providers.begin(), providers.end(), providerName) != providers.end();
        } catch (const Ort::Exception& e) {
            LOG_WARN("Delimiter: failed to query ORT providers: {}", e.what());
            return false;
        } catch (const std::exception& e) {
            LOG_WARN("Delimiter: failed to query ORT providers: {}", e.what());
            return false;
        }
    }

    bool appendProvider(Ort::SessionOptions& options, const Provider& provider) {
        if (provider.type == ProviderType::Cpu) {
            return true;  // CPU is the default provider.
        }

        OrtStatus* status = nullptr;
        if (provider.type == ProviderType::Cuda) {
            // ORT 1.17+ requires provider option struct instead of device id
#if ORT_API_VERSION >= 17
            OrtCUDAProviderOptions cudaOptions{};
            cudaOptions.device_id = 0;
            status =
                Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA(options, &cudaOptions);
#else
            status = Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA(options, 0);
#endif
            if (status) {
                initError_ = statusMessage(status, "CUDA provider init failed: ");
                return false;
            }
            return true;
        }

// ORT 1.17+ requires provider option struct instead of device id
#if ORT_API_VERSION >= 17
        OrtTensorRTProviderOptions trtOptions{};
        trtOptions.device_id = 0;
        status = Ort::GetApi().SessionOptionsAppendExecutionProvider_TensorRT(options, &trtOptions);
#else
        status = Ort::GetApi().SessionOptionsAppendExecutionProvider_TensorRT(options, 0);
#endif
        if (status) {
            initError_ = statusMessage(status, "TensorRT provider init failed: ");
            return false;
        }
        return true;
    }

    void loadIoNames() {
        if (!session_) {
            return;
        }

        Ort::AllocatorWithDefaultOptions allocator;
        if (session_->GetInputCount() == 0) {
            initError_ = "ORT model has no inputs";
            session_.reset();
            return;
        }

        inputName_ = session_->GetInputNameAllocated(0, allocator).get();

        size_t outputCount = session_->GetOutputCount();
        if (outputCount == 0) {
            initError_ = "ORT model has no outputs";
            session_.reset();
            return;
        }

        outputNames_.clear();
        outputNamePtrs_.clear();
        for (size_t i = 0; i < outputCount; ++i) {
            std::string name = session_->GetOutputNameAllocated(i, allocator).get();
            outputNames_.push_back(std::move(name));
        }
        for (const auto& name : outputNames_) {
            outputNamePtrs_.push_back(name.c_str());
        }
        initError_.clear();
    }

    InferenceResult extractOutputs(const Ort::Value& value, std::vector<float>& outLeft,
                                   std::vector<float>& outRight) const {
        if (!value.IsTensor()) {
            return {InferenceStatus::Error, "ORT output is not a tensor"};
        }

        auto info = value.GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        const float* data = value.GetTensorData<float>();
        if (!data) {
            return {InferenceStatus::Error, "ORT output tensor is empty"};
        }

        auto elementCount = static_cast<std::size_t>(info.GetElementCount());
        if (elementCount == 0) {
            return {InferenceStatus::Error, "ORT output tensor has zero elements"};
        }

        auto copyChannelFirst = [&](std::size_t frames) {
            if (frames == 0 || frames * 2 != elementCount) {
                return InferenceResult{InferenceStatus::Error, "ORT output shape mismatch"};
            }
            outLeft.assign(data, data + frames);
            outRight.assign(data + frames, data + frames * 2);
            return InferenceResult{InferenceStatus::Ok, ""};
        };

        if (shape.size() == 3 && (shape[0] == 1 || shape[0] == -1) &&
            (shape[1] == 2 || shape[1] == -1) && shape[2] > 0) {
            return copyChannelFirst(static_cast<std::size_t>(shape[2]));
        }

        if (shape.size() == 2 && shape[0] == 2 && shape[1] > 0) {
            return copyChannelFirst(static_cast<std::size_t>(shape[1]));
        }

        if (shape.size() == 2 && shape[1] == 2 && shape[0] > 0) {
            std::size_t frames = static_cast<std::size_t>(shape[0]);
            if (frames * 2 != elementCount) {
                return {InferenceStatus::Error, "ORT output shape mismatch"};
            }
            outLeft.resize(frames);
            outRight.resize(frames);
            for (std::size_t i = 0; i < frames; ++i) {
                outLeft[i] = data[i * 2];
                outRight[i] = data[i * 2 + 1];
            }
            return {InferenceStatus::Ok, ""};
        }

        return {InferenceStatus::Error, "Unsupported ORT output shape"};
    }

    AppConfig::DelimiterConfig config_;
    uint32_t expectedSampleRate_ = kDelimiterRate44k;
    Ort::MemoryInfo memoryInfo_;
    std::unique_ptr<Ort::Session> session_;
    std::string inputName_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> outputNamePtrs_;
    std::vector<float> inputBuffer_;
    std::string initError_;
};

#else  // DELIMITER_ENABLE_ORT

class OrtInferenceBackend final : public InferenceBackend {
   public:
    explicit OrtInferenceBackend(const AppConfig::DelimiterConfig& config)
        : expectedSampleRate_(config.expectedSampleRate) {}

    const char* name() const override {
        return "ort";
    }

    uint32_t expectedSampleRate() const override {
        return expectedSampleRate_;
    }

    InferenceResult process(const StereoPlanarView& input, std::vector<float>& outLeft,
                            std::vector<float>& outRight) override {
        outLeft.clear();
        outRight.clear();
        if (!input.valid() || input.frames == 0) {
            return {InferenceStatus::InvalidConfig, "invalid input buffer"};
        }
        return {InferenceStatus::Unsupported,
                "ONNX Runtime backend is not enabled at build time (rebuild with "
                "DELIMITER_ENABLE_ORT=ON)"};
    }

    void reset() override {}

   private:
    uint32_t expectedSampleRate_;
};

#endif  // DELIMITER_ENABLE_ORT

}  // namespace

std::unique_ptr<InferenceBackend> createDelimiterInferenceBackend(
    const AppConfig::DelimiterConfig& config) {
    if (!config.enabled) {
        return std::make_unique<BypassInferenceBackend>(config.expectedSampleRate);
    }

    std::string backend = toLower(config.backend);

    if (backend == "bypass") {
        return std::make_unique<BypassInferenceBackend>(config.expectedSampleRate);
    }

    if (backend == "ort" || backend == "onnx" || backend == "onnxruntime") {
        return std::make_unique<OrtInferenceBackend>(config);
    }

    LOG_WARN("Delimiter: Unknown backend '{}' (falling back to bypass)", config.backend);
    return std::make_unique<BypassInferenceBackend>(config.expectedSampleRate);
}

}  // namespace delimiter
