#include <lv2/core/lv2.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "../../include/convolution_engine.h"
#include "gpu_upsampler_lv2.h"

// Use ConvolutionEngine namespace
using ConvolutionEngine::GPUUpsampler;

// Plugin instance structure (C++ struct)
struct GPUUpsamplerLV2 {
    // Port buffers
    const float* input_left;
    const float* input_right;
    float* output_left;
    float* output_right;

    // GPU upsampler engine
    GPUUpsampler* upsampler;

    // Sample rate
    double sample_rate;

    // Processing buffers
    std::vector<float>* output_buffer_left;
    std::vector<float>* output_buffer_right;

    // Filter path
    const char* filter_path;

    // Upsample ratio
    int upsample_ratio;
};

// Instantiate the plugin
static LV2_Handle
instantiate(const LV2_Descriptor* descriptor,
            double rate,
            const char* bundle_path,
            const LV2_Feature* const* features)
{
    (void)descriptor;
    (void)bundle_path;
    (void)features;

    GPUUpsamplerLV2* plugin = new GPUUpsamplerLV2();
    if (!plugin) {
        return nullptr;
    }

    plugin->input_left = nullptr;
    plugin->input_right = nullptr;
    plugin->output_left = nullptr;
    plugin->output_right = nullptr;
    plugin->upsampler = nullptr;
    plugin->sample_rate = rate;
    plugin->output_buffer_left = new std::vector<float>();
    plugin->output_buffer_right = new std::vector<float>();

    // Default filter path (will be configurable later)
    plugin->filter_path = "/home/michihito/Working/gpu_os/data/coefficients/filter_1m_min_phase.bin";
    plugin->upsample_ratio = 16;

    return (LV2_Handle)plugin;
}

// Connect port to buffer
static void
connect_port(LV2_Handle instance,
             uint32_t port,
             void* data)
{
    GPUUpsamplerLV2* plugin = (GPUUpsamplerLV2*)instance;

    switch (port) {
        case PORT_AUDIO_IN_L:
            plugin->input_left = (const float*)data;
            break;
        case PORT_AUDIO_IN_R:
            plugin->input_right = (const float*)data;
            break;
        case PORT_AUDIO_OUT_L:
            plugin->output_left = (float*)data;
            break;
        case PORT_AUDIO_OUT_R:
            plugin->output_right = (float*)data;
            break;
    }
}

// Activate the plugin (prepare for processing)
static void
activate(LV2_Handle instance)
{
    GPUUpsamplerLV2* plugin = (GPUUpsamplerLV2*)instance;

    // Initialize GPU upsampler engine
    try {
        plugin->upsampler = new GPUUpsampler();

        if (!plugin->upsampler->initialize(plugin->filter_path, plugin->upsample_ratio, 8192)) {
            delete plugin->upsampler;
            plugin->upsampler = nullptr;
            return;
        }
    } catch (...) {
        if (plugin->upsampler) {
            delete plugin->upsampler;
            plugin->upsampler = nullptr;
        }
    }
}

// Process audio (run function)
static void
run(LV2_Handle instance, uint32_t n_samples)
{
    GPUUpsamplerLV2* plugin = (GPUUpsamplerLV2*)instance;

    // Safety check
    if (!plugin->upsampler || !plugin->input_left || !plugin->input_right ||
        !plugin->output_left || !plugin->output_right) {
        // Passthrough mode if upsampler not initialized
        if (plugin->input_left && plugin->output_left) {
            memcpy(plugin->output_left, plugin->input_left, n_samples * sizeof(float));
        }
        if (plugin->input_right && plugin->output_right) {
            memcpy(plugin->output_right, plugin->input_right, n_samples * sizeof(float));
        }
        return;
    }

    // Process stereo audio through GPU upsampler
    bool success = plugin->upsampler->processStereo(
        plugin->input_left,
        plugin->input_right,
        n_samples,
        *plugin->output_buffer_left,
        *plugin->output_buffer_right
    );

    if (!success) {
        // On error, output silence
        memset(plugin->output_left, 0, n_samples * plugin->upsample_ratio * sizeof(float));
        memset(plugin->output_right, 0, n_samples * plugin->upsample_ratio * sizeof(float));
        return;
    }

    // Copy upsampled data to output ports
    // Note: LV2 host expects the same sample rate, so we need to handle this carefully
    // For now, we'll do a simple decimation (take every Nth sample)
    // In production, this should be handled by PipeWire's sample rate negotiation
    size_t output_frames = plugin->output_buffer_left->size();
    for (uint32_t i = 0; i < n_samples && i < output_frames; i++) {
        plugin->output_left[i] = (*plugin->output_buffer_left)[i];
        plugin->output_right[i] = (*plugin->output_buffer_right)[i];
    }
}

// Deactivate the plugin
static void
deactivate(LV2_Handle instance)
{
    GPUUpsamplerLV2* plugin = (GPUUpsamplerLV2*)instance;

    if (plugin->upsampler) {
        delete plugin->upsampler;
        plugin->upsampler = nullptr;
    }
}

// Cleanup and free the plugin instance
static void
cleanup(LV2_Handle instance)
{
    GPUUpsamplerLV2* plugin = (GPUUpsamplerLV2*)instance;

    if (plugin->output_buffer_left) {
        delete plugin->output_buffer_left;
    }
    if (plugin->output_buffer_right) {
        delete plugin->output_buffer_right;
    }

    delete plugin;
}

// LV2 descriptor
static const LV2_Descriptor descriptor = {
    GPU_UPSAMPLER_URI,
    instantiate,
    connect_port,
    activate,
    run,
    deactivate,
    cleanup,
    nullptr  // extension_data
};

// Entry point: return descriptor
LV2_SYMBOL_EXPORT
const LV2_Descriptor*
lv2_descriptor(uint32_t index)
{
    return (index == 0) ? &descriptor : nullptr;
}
