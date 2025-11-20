#include <lv2/core/lv2.h>
#include <lv2/worker/worker.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "../../include/convolution_engine.h"
#include "gpu_upsampler_lv2.h"

// Work message types
enum WorkMessageType {
    WORK_MSG_PROCESS_AUDIO = 0
};

// Work message structure (sent from run() to work())
// Note: No pointers - worker accesses plugin instance buffers directly
struct WorkMessage {
    WorkMessageType type;
    uint32_t n_samples;
};

// Response message structure (sent from work() to work_response())
struct ResponseMessage {
    WorkMessageType type;
    uint32_t output_samples;
    bool success;
};

// Use ConvolutionEngine namespace
using ConvolutionEngine::GPUUpsampler;

// Plugin instance structure (C++ struct)
struct GPUUpsamplerLV2 {
    // Port buffers (connected by host)
    const float* input_left;
    const float* input_right;
    float* output_left;
    float* output_right;
    float* latency_port;

    // GPU upsampler engine
    GPUUpsampler* upsampler;

    // Sample rate
    double sample_rate;

    // Worker input buffers (copied from port buffers in run())
    std::vector<float>* worker_input_left;
    std::vector<float>* worker_input_right;

    // Worker output buffers (filled by GPU processing in work())
    std::vector<float>* worker_output_left;
    std::vector<float>* worker_output_right;

    // Worker state
    bool worker_processing;  // True if worker is currently processing
    uint32_t output_read_pos;  // Position in output buffer for consumption

    // Filter path
    const char* filter_path;

    // Upsample ratio
    int upsample_ratio;

    // Worker interface
    LV2_Worker_Schedule* schedule;
};

// Instantiate the plugin
static LV2_Handle
instantiate(const LV2_Descriptor* descriptor,
            double rate,
            const char* bundle_path,
            const LV2_Feature* const* features)
{
    (void)descriptor;

    GPUUpsamplerLV2* plugin = new GPUUpsamplerLV2();
    if (!plugin) {
        return nullptr;
    }

    plugin->input_left = nullptr;
    plugin->input_right = nullptr;
    plugin->output_left = nullptr;
    plugin->output_right = nullptr;
    plugin->latency_port = nullptr;
    plugin->upsampler = nullptr;
    plugin->sample_rate = rate;

    // Initialize worker buffers
    plugin->worker_input_left = new std::vector<float>();
    plugin->worker_input_right = new std::vector<float>();
    plugin->worker_output_left = new std::vector<float>();
    plugin->worker_output_right = new std::vector<float>();
    plugin->worker_processing = false;
    plugin->output_read_pos = 0;

    plugin->schedule = nullptr;

    // Construct filter path from bundle path (installed with plugin)
    std::string filter_path_str = std::string(bundle_path) + "/filter_1m_min_phase.bin";
    // Allocate persistent string (plugin lifetime)
    char* filter_path_copy = new char[filter_path_str.length() + 1];
    std::strcpy(filter_path_copy, filter_path_str.c_str());
    plugin->filter_path = filter_path_copy;
    plugin->upsample_ratio = 16;

    // Get Worker schedule interface from host
    for (int i = 0; features[i]; i++) {
        if (!strcmp(features[i]->URI, LV2_WORKER__schedule)) {
            plugin->schedule = (LV2_Worker_Schedule*)features[i]->data;
        }
    }

    if (!plugin->schedule) {
        // Worker interface is required
        delete[] filter_path_copy;
        delete plugin->worker_input_left;
        delete plugin->worker_input_right;
        delete plugin->worker_output_left;
        delete plugin->worker_output_right;
        delete plugin;
        return nullptr;
    }

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
        case PORT_LATENCY:
            plugin->latency_port = (float*)data;
            break;
    }
}

// Activate the plugin (prepare for processing)
static void
activate(LV2_Handle instance)
{
    GPUUpsamplerLV2* plugin = (GPUUpsamplerLV2*)instance;

    // Clean up existing instance if present (prevent memory leak on re-activation)
    if (plugin->upsampler) {
        delete plugin->upsampler;
        plugin->upsampler = nullptr;
    }

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

    // Report latency to host
    // Latency components:
    // 1. FFT overlap (filterTaps - 1 = 999,999 samples at input rate)
    // 2. Block size (8192 samples at input rate)
    // Total: ~1,008,191 samples @ 44.1kHz = ~22.86 seconds
    //
    // Note: This is the MINIMUM latency due to minimum-phase FIR filter design.
    // The 1M tap filter concentrates energy at t>=0 but still requires full convolution.
    if (plugin->latency_port) {
        const int filter_taps = 1000000;  // 1M tap filter
        const int overlap_latency = filter_taps - 1;  // 999,999
        const int block_latency = 8192;
        const int total_latency = overlap_latency + block_latency;
        *plugin->latency_port = static_cast<float>(total_latency);
    }
}

// Process audio (run function)
static void
run(LV2_Handle instance, uint32_t n_samples)
{
    GPUUpsamplerLV2* plugin = (GPUUpsamplerLV2*)instance;

    // Safety check
    if (!plugin->upsampler || !plugin->input_left || !plugin->input_right ||
        !plugin->output_left || !plugin->output_right || !plugin->schedule) {
        // Passthrough mode if upsampler not initialized
        if (plugin->input_left && plugin->output_left) {
            memcpy(plugin->output_left, plugin->input_left, n_samples * sizeof(float));
        }
        if (plugin->input_right && plugin->output_right) {
            memcpy(plugin->output_right, plugin->input_right, n_samples * sizeof(float));
        }
        return;
    }

    // Step 1: Copy input to worker buffers (if worker is not currently processing)
    if (!plugin->worker_processing) {
        plugin->worker_input_left->assign(plugin->input_left, plugin->input_left + n_samples);
        plugin->worker_input_right->assign(plugin->input_right, plugin->input_right + n_samples);

        // Step 2: Schedule GPU processing in worker thread
        WorkMessage msg;
        msg.type = WORK_MSG_PROCESS_AUDIO;
        msg.n_samples = n_samples;

        // Mark as processing
        plugin->worker_processing = true;

        // Schedule work (non-blocking, returns immediately)
        plugin->schedule->schedule_work(
            plugin->schedule->handle,
            sizeof(WorkMessage),
            &msg
        );
    }

    // Step 3: Output from previous worker results
    // Note: Initial latency until first result is ready
    size_t available = plugin->worker_output_left->size() - plugin->output_read_pos;

    if (available >= n_samples) {
        // Enough data available - copy to output
        for (uint32_t i = 0; i < n_samples; i++) {
            plugin->output_left[i] = (*plugin->worker_output_left)[plugin->output_read_pos + i];
            plugin->output_right[i] = (*plugin->worker_output_right)[plugin->output_read_pos + i];
        }
        plugin->output_read_pos += n_samples;
    } else {
        // Not enough data - output silence (underrun)
        memset(plugin->output_left, 0, n_samples * sizeof(float));
        memset(plugin->output_right, 0, n_samples * sizeof(float));
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

// Worker thread function: performs GPU processing
static LV2_Worker_Status
work(LV2_Handle instance,
     LV2_Worker_Respond_Function respond,
     LV2_Worker_Respond_Handle handle,
     uint32_t size,
     const void* data)
{
    GPUUpsamplerLV2* plugin = (GPUUpsamplerLV2*)instance;

    if (size != sizeof(WorkMessage)) {
        return LV2_WORKER_ERR_UNKNOWN;
    }

    const WorkMessage* msg = (const WorkMessage*)data;

    if (msg->type != WORK_MSG_PROCESS_AUDIO) {
        return LV2_WORKER_ERR_UNKNOWN;
    }

    // Perform GPU processing in worker thread (non-realtime)
    // Access input from worker_input buffers (safe - copied in run())
    bool success = false;
    if (plugin->upsampler && plugin->worker_input_left && plugin->worker_input_right) {
        success = plugin->upsampler->processStereo(
            plugin->worker_input_left->data(),
            plugin->worker_input_right->data(),
            msg->n_samples,
            *plugin->worker_output_left,
            *plugin->worker_output_right
        );
    }

    // Prepare response message
    ResponseMessage response;
    response.type = WORK_MSG_PROCESS_AUDIO;
    response.success = success;
    response.output_samples = success ? plugin->worker_output_left->size() : 0;

    // Send response back to audio thread
    respond(handle, sizeof(ResponseMessage), &response);

    return LV2_WORKER_SUCCESS;
}

// Audio thread function: receives response from worker
static LV2_Worker_Status
work_response(LV2_Handle instance,
              uint32_t size,
              const void* data)
{
    GPUUpsamplerLV2* plugin = (GPUUpsamplerLV2*)instance;

    if (size != sizeof(ResponseMessage)) {
        return LV2_WORKER_ERR_UNKNOWN;
    }

    const ResponseMessage* response = (const ResponseMessage*)data;

    // Mark worker as finished
    plugin->worker_processing = false;

    // Reset read position for new output data
    if (response->success && response->output_samples > 0) {
        plugin->output_read_pos = 0;
        // Worker output buffers are already filled by work()
    } else {
        // GPU processing failed - clear output buffers
        plugin->worker_output_left->clear();
        plugin->worker_output_right->clear();
        plugin->output_read_pos = 0;
    }

    return LV2_WORKER_SUCCESS;
}

// Cleanup and free the plugin instance
static void
cleanup(LV2_Handle instance)
{
    GPUUpsamplerLV2* plugin = (GPUUpsamplerLV2*)instance;

    // Free worker buffers
    if (plugin->worker_input_left) {
        delete plugin->worker_input_left;
    }
    if (plugin->worker_input_right) {
        delete plugin->worker_input_right;
    }
    if (plugin->worker_output_left) {
        delete plugin->worker_output_left;
    }
    if (plugin->worker_output_right) {
        delete plugin->worker_output_right;
    }

    // Free filter path string allocated in instantiate()
    if (plugin->filter_path) {
        delete[] plugin->filter_path;
    }

    delete plugin;
}

// Worker interface structure
static const LV2_Worker_Interface worker_interface = {
    work,
    work_response,
    nullptr  // end_run (optional)
};

// Extension data function: exposes Worker interface to host
static const void*
extension_data(const char* uri)
{
    if (!strcmp(uri, LV2_WORKER__interface)) {
        return &worker_interface;
    }
    return nullptr;
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
    extension_data
};

// Entry point: return descriptor
LV2_SYMBOL_EXPORT
const LV2_Descriptor*
lv2_descriptor(uint32_t index)
{
    return (index == 0) ? &descriptor : nullptr;
}
