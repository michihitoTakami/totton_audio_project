#include "audio_utils.h"
#include "base64.h"
#include "config_loader.h"
#include "convolution_engine.h"
#include "crossfeed_engine.h"
#include "daemon_constants.h"
#include "eq_parser.h"
#include "eq_to_fir.h"
#include "soft_mute.h"

#include <algorithm>
#include <alsa/asoundlib.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <zmq.hpp>

// PID file path (also serves as lock file)
constexpr const char* PID_FILE_PATH = "/tmp/gpu_upsampler_alsa.pid";

// Stats file path (JSON format for Web API)
constexpr const char* STATS_FILE_PATH = "/tmp/gpu_upsampler_stats.json";

// ZeroMQ IPC socket path (matching Python client)
constexpr const char* ZEROMQ_IPC_PATH = "ipc:///tmp/gpu_os.sock";

// Default configuration values (using common constants)
using namespace DaemonConstants;
constexpr const char* DEFAULT_ALSA_DEVICE = "hw:USB";
constexpr const char* DEFAULT_FILTER_PATH = "data/coefficients/filter_1m_min_phase.bin";
constexpr const char* CONFIG_FILE_PATH = DEFAULT_CONFIG_FILE;

// Runtime configuration (loaded from config.json)
static AppConfig g_config;

// Global state
static std::atomic<bool> g_running{true};
static std::atomic<bool> g_reload_requested{false};
static std::atomic<bool> g_main_loop_running{false};  // True when pw_main_loop_run() is active
static std::atomic<bool> g_zmq_bind_failed{false};    // True if ZeroMQ bind failed
static ConvolutionEngine::GPUUpsampler* g_upsampler = nullptr;
static struct pw_main_loop* g_pw_loop = nullptr;  // For SIGHUP handler

// Statistics (atomic for thread-safe access from stats writer)
static std::atomic<size_t> g_clip_count{0};
static std::atomic<size_t> g_total_samples{0};

// Soft Mute controller for glitch-free shutdown (50ms fade at output sample rate)
static SoftMute::Controller* g_soft_mute = nullptr;

// Audio buffer for thread communication
static std::mutex g_buffer_mutex;
static std::condition_variable g_buffer_cv;
static std::vector<float> g_output_buffer_left;
static std::vector<float> g_output_buffer_right;
static size_t g_output_read_pos = 0;

// Streaming input accumulation buffers
static std::vector<float> g_stream_input_left;
static std::vector<float> g_stream_input_right;
static size_t g_stream_accumulated_left = 0;
static size_t g_stream_accumulated_right = 0;

// Crossfeed (HRTF) processor
static CrossfeedEngine::HRTFProcessor* g_hrtf_processor = nullptr;
static std::atomic<bool> g_crossfeed_enabled{false};  // Atomic for quick check
static std::vector<float> g_cf_stream_input_left;
static std::vector<float> g_cf_stream_input_right;
static size_t g_cf_stream_accumulated_left = 0;
static size_t g_cf_stream_accumulated_right = 0;
static std::mutex g_crossfeed_mutex;  // Protects HRTFProcessor and CF buffers

// ========== PID File Lock (flock-based) ==========

// File descriptor for the PID lock file (kept open while running)
static int g_pid_lock_fd = -1;

// Read PID from lock file (for display purposes)
static pid_t read_pid_from_lockfile() {
    std::ifstream pidfile(PID_FILE_PATH);
    if (!pidfile.is_open())
        return 0;
    pid_t pid = 0;
    pidfile >> pid;
    return pid;
}

// Acquire exclusive lock on PID file using flock()
// Returns true if lock acquired, false if another instance is running
static bool acquire_pid_lock() {
    // Open or create the lock file
    g_pid_lock_fd = open(PID_FILE_PATH, O_RDWR | O_CREAT, 0644);
    if (g_pid_lock_fd < 0) {
        std::cerr << "Error: Cannot open PID file: " << PID_FILE_PATH << " (" << strerror(errno)
                  << ")" << std::endl;
        return false;
    }

    // Try to acquire exclusive lock (non-blocking)
    if (flock(g_pid_lock_fd, LOCK_EX | LOCK_NB) < 0) {
        if (errno == EWOULDBLOCK) {
            // Another process holds the lock
            pid_t existing_pid = read_pid_from_lockfile();
            std::cerr << "Error: Another instance is already running";
            if (existing_pid > 0) {
                std::cerr << " (PID: " << existing_pid << ")";
            }
            std::cerr << std::endl;
            std::cerr << "       Use './scripts/daemon.sh stop' to stop it." << std::endl;
        } else {
            std::cerr << "Error: Cannot lock PID file: " << strerror(errno) << std::endl;
        }
        close(g_pid_lock_fd);
        g_pid_lock_fd = -1;
        return false;
    }

    // Lock acquired - write our PID to the file
    if (ftruncate(g_pid_lock_fd, 0) < 0) {
        std::cerr << "Warning: Cannot truncate PID file" << std::endl;
    }
    dprintf(g_pid_lock_fd, "%d\n", getpid());
    fsync(g_pid_lock_fd);  // Ensure PID is written to disk

    return true;
}

// Release PID lock and remove file
// Note: Lock is automatically released when process exits (even on crash)
static void release_pid_lock() {
    if (g_pid_lock_fd >= 0) {
        flock(g_pid_lock_fd, LOCK_UN);
        close(g_pid_lock_fd);
        g_pid_lock_fd = -1;
    }
    // Remove the PID file on clean shutdown
    unlink(PID_FILE_PATH);
    // Remove the stats file on clean shutdown
    unlink(STATS_FILE_PATH);
}

// ========== Statistics File ==========

// Write statistics to JSON file for Web API consumption
static void write_stats_file() {
    size_t clips = g_clip_count.load(std::memory_order_relaxed);
    size_t total = g_total_samples.load(std::memory_order_relaxed);
    double clip_rate = (total > 0) ? (static_cast<double>(clips) / total) : 0.0;
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    // Calculate actual output rate from running config
    int input_rate = g_config.inputSampleRate;
    int upsample_ratio = g_config.upsampleRatio;
    int output_rate = input_rate * upsample_ratio;

    // Write to temp file and rename for atomic update
    std::string tmp_path = std::string(STATS_FILE_PATH) + ".tmp";
    std::ofstream ofs(tmp_path);
    if (ofs) {
        ofs << "{\n"
            << "  \"clip_count\": " << clips << ",\n"
            << "  \"total_samples\": " << total << ",\n"
            << "  \"clip_rate\": " << clip_rate << ",\n"
            << "  \"input_rate\": " << input_rate << ",\n"
            << "  \"output_rate\": " << output_rate << ",\n"
            << "  \"upsample_ratio\": " << upsample_ratio << ",\n"
            << "  \"last_updated\": " << epoch << "\n"
            << "}\n";
        ofs.close();
        // Atomic rename
        std::rename(tmp_path.c_str(), STATS_FILE_PATH);
    }
}

// ========== Configuration ==========

static void print_config_summary(const AppConfig& cfg) {
    int outputRate = cfg.inputSampleRate * cfg.upsampleRatio;
    std::cout << "Config:" << std::endl;
    std::cout << "  ALSA device:    " << cfg.alsaDevice << std::endl;
    std::cout << "  Input rate:     " << cfg.inputSampleRate << " Hz" << std::endl;
    std::cout << "  Output rate:    " << outputRate << " Hz (" << (outputRate / 1000.0) << " kHz)"
              << std::endl;
    std::cout << "  Buffer size:    " << cfg.bufferSize << std::endl;
    std::cout << "  Period size:    " << cfg.periodSize << std::endl;
    std::cout << "  Upsample ratio: " << cfg.upsampleRatio << std::endl;
    std::cout << "  Block size:     " << cfg.blockSize << std::endl;
    std::cout << "  Gain:           " << cfg.gain << std::endl;
    std::cout << "  Filter path:    " << cfg.filterPath << std::endl;
    std::cout << "  EQ enabled:     " << (cfg.eqEnabled ? "yes" : "no") << std::endl;
    if (cfg.eqEnabled && !cfg.eqProfilePath.empty()) {
        std::cout << "  EQ profile:     " << cfg.eqProfilePath << std::endl;
    }
}

static void reset_runtime_state() {
    g_output_buffer_left.clear();
    g_output_buffer_right.clear();
    g_output_read_pos = 0;
    g_stream_input_left.clear();
    g_stream_input_right.clear();
    g_stream_accumulated_left = 0;
    g_stream_accumulated_right = 0;

    // Reset crossfeed streaming buffers
    g_cf_stream_input_left.clear();
    g_cf_stream_input_right.clear();
    g_cf_stream_accumulated_left = 0;
    g_cf_stream_accumulated_right = 0;
}

static void load_runtime_config() {
    AppConfig loaded;
    bool found = loadAppConfig(CONFIG_FILE_PATH, loaded);
    g_config = loaded;

    if (g_config.alsaDevice.empty())
        g_config.alsaDevice = DEFAULT_ALSA_DEVICE;
    if (g_config.filterPath.empty())
        g_config.filterPath = DEFAULT_FILTER_PATH;
    if (g_config.upsampleRatio <= 0)
        g_config.upsampleRatio = DEFAULT_UPSAMPLE_RATIO;
    if (g_config.blockSize <= 0)
        g_config.blockSize = DEFAULT_BLOCK_SIZE;
    if (g_config.bufferSize <= 0)
        g_config.bufferSize = 262144;
    if (g_config.periodSize <= 0)
        g_config.periodSize = 32768;
    if (g_config.inputSampleRate != 44100 && g_config.inputSampleRate != 48000) {
        g_config.inputSampleRate = DEFAULT_INPUT_SAMPLE_RATE;
    }

    if (!found) {
        std::cout << "Config: Using defaults (no config.json found)" << std::endl;
    }
    print_config_summary(g_config);
}

// ========== ZeroMQ Command Listener ==========

// Build stats JSON response for ZeroMQ STATS command
static std::string build_stats_json() {
    size_t clips = g_clip_count.load(std::memory_order_relaxed);
    size_t total = g_total_samples.load(std::memory_order_relaxed);
    double clip_rate = (total > 0) ? (static_cast<double>(clips) / total) : 0.0;

    // Get phase type from upsampler (or config default if not initialized)
    std::string phaseTypeStr = "minimum";
    if (g_upsampler) {
        phaseTypeStr = (g_upsampler->getPhaseType() == PhaseType::Minimum) ? "minimum" : "linear";
    } else {
        phaseTypeStr = (g_config.phaseType == PhaseType::Minimum) ? "minimum" : "linear";
    }

    std::ostringstream oss;
    oss << "{"
        << "\"clip_count\":" << clips << ","
        << "\"total_samples\":" << total << ","
        << "\"clip_rate\":" << clip_rate << ","
        << "\"eq_enabled\":" << (g_config.eqEnabled ? "true" : "false") << ","
        << "\"phase_type\":\"" << phaseTypeStr << "\","
        << "\"input_rate\":" << g_config.inputSampleRate << ","
        << "\"upsample_ratio\":" << g_config.upsampleRatio << "}";
    return oss.str();
}

// ZeroMQ listener thread - handles PING, RELOAD, STATS commands from Python/FastAPI
static void zeromq_listener_thread() {
    try {
        zmq::context_t context(1);
        zmq::socket_t socket(context, zmq::socket_type::rep);

        // Set timeouts for graceful shutdown
        socket.set(zmq::sockopt::rcvtimeo, 1000);  // 1 second receive timeout
        socket.set(zmq::sockopt::linger, 0);

        // Remove stale socket file if exists (from previous crash/unclean shutdown)
        std::string sock_path = std::string(ZEROMQ_IPC_PATH).substr(6);  // Remove "ipc://"
        unlink(sock_path.c_str());  // Ignore error if file doesn't exist

        socket.bind(ZEROMQ_IPC_PATH);
        std::cout << "ZeroMQ: Listening on " << ZEROMQ_IPC_PATH << std::endl;

        while (g_running.load()) {
            zmq::message_t request;

            // Receive with timeout (non-blocking check for shutdown)
            auto result = socket.recv(request, zmq::recv_flags::none);
            if (!result) {
                // Timeout - check if we should continue
                continue;
            }

            std::string cmd(static_cast<char*>(request.data()), request.size());
            std::string response;

            // Process command
            if (cmd == "PING") {
                response = "OK";
            } else if (cmd == "RELOAD") {
                // Request config reload (same as SIGHUP)
                g_reload_requested = true;
                // Start fade-out for glitch-free reload
                if (g_soft_mute) {
                    g_soft_mute->startFadeOut();
                }
                // Only quit main loop if it's actually running
                // If not running yet, main() will check g_reload_requested before starting
                if (g_main_loop_running.load() && g_pw_loop) {
                    pw_main_loop_quit(g_pw_loop);
                }
                response = "OK";
            } else if (cmd == "STATS") {
                response = "OK:" + build_stats_json();
            } else if (cmd == "CROSSFEED_ENABLE") {
                std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
                if (g_hrtf_processor) {
                    // Reset streaming state to avoid stale data from previous session
                    g_hrtf_processor->resetStreaming();
                    g_cf_stream_input_left.clear();
                    g_cf_stream_input_right.clear();
                    g_cf_stream_accumulated_left = 0;
                    g_cf_stream_accumulated_right = 0;

                    g_hrtf_processor->setEnabled(true);
                    g_crossfeed_enabled.store(true);
                    response = "OK:Crossfeed enabled";
                } else {
                    response = "ERR:HRTF processor not initialized";
                }
            } else if (cmd == "CROSSFEED_DISABLE") {
                std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
                g_crossfeed_enabled.store(false);
                if (g_hrtf_processor) {
                    g_hrtf_processor->setEnabled(false);
                }
                response = "OK:Crossfeed disabled";
            } else if (cmd == "CROSSFEED_STATUS") {
                std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
                bool enabled = g_crossfeed_enabled.load();
                bool initialized = (g_hrtf_processor != nullptr);
                response = "OK:{\"enabled\":" + std::string(enabled ? "true" : "false") +
                           ",\"initialized\":" + std::string(initialized ? "true" : "false") + "}";
            }
            // JSON format commands for crossfeed (#150)
            else if (cmd.find("{") == 0) {
                // Parse JSON command
                try {
                    nlohmann::json j = nlohmann::json::parse(cmd);
                    std::string cmdType = j.value("cmd", "");

                    if (cmdType == "CROSSFEED_ENABLE") {
                        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
                        if (g_hrtf_processor) {
                            g_hrtf_processor->resetStreaming();
                            g_cf_stream_input_left.clear();
                            g_cf_stream_input_right.clear();
                            g_cf_stream_accumulated_left = 0;
                            g_cf_stream_accumulated_right = 0;
                            g_hrtf_processor->setEnabled(true);
                            g_crossfeed_enabled.store(true);
                            nlohmann::json resp;
                            resp["status"] = "ok";
                            resp["message"] = "Crossfeed enabled";
                            response = resp.dump();
                        } else {
                            nlohmann::json resp;
                            resp["status"] = "error";
                            resp["error_code"] = "CROSSFEED_NOT_INITIALIZED";
                            resp["message"] = "HRTF processor not initialized";
                            response = resp.dump();
                        }
                    } else if (cmdType == "CROSSFEED_DISABLE") {
                        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
                        g_crossfeed_enabled.store(false);
                        if (g_hrtf_processor) {
                            g_hrtf_processor->setEnabled(false);
                        }
                        nlohmann::json resp;
                        resp["status"] = "ok";
                        resp["message"] = "Crossfeed disabled";
                        response = resp.dump();
                    } else if (cmdType == "CROSSFEED_SET_COMBINED") {
                        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
                        if (!g_hrtf_processor) {
                            nlohmann::json resp;
                            resp["status"] = "error";
                            resp["error_code"] = "CROSSFEED_NOT_INITIALIZED";
                            resp["message"] = "HRTF processor not initialized";
                            response = resp.dump();
                        } else if (!j.contains("params")) {
                            nlohmann::json resp;
                            resp["status"] = "error";
                            resp["error_code"] = "IPC_INVALID_PARAMS";
                            resp["message"] = "Missing params field";
                            response = resp.dump();
                        } else {
                            auto params = j["params"];
                            std::string rateFamily = params.value("rate_family", "");
                            std::string combinedLL = params.value("combined_ll", "");
                            std::string combinedLR = params.value("combined_lr", "");
                            std::string combinedRL = params.value("combined_rl", "");
                            std::string combinedRR = params.value("combined_rr", "");

                            if (rateFamily.empty() || combinedLL.empty() || combinedLR.empty() ||
                                combinedRL.empty() || combinedRR.empty()) {
                                nlohmann::json resp;
                                resp["status"] = "error";
                                resp["error_code"] = "IPC_INVALID_PARAMS";
                                resp["message"] = "Missing required filter data";
                                response = resp.dump();
                            } else if (rateFamily != "44k" && rateFamily != "48k") {
                                nlohmann::json resp;
                                resp["status"] = "error";
                                resp["error_code"] = "CROSSFEED_INVALID_RATE_FAMILY";
                                resp["message"] =
                                    "Invalid rate family: " + rateFamily + " (expected 44k or 48k)";
                                response = resp.dump();
                            } else {
                                // Decode Base64 filter data
                                auto decodedLL = Base64::decode(combinedLL);
                                auto decodedLR = Base64::decode(combinedLR);
                                auto decodedRL = Base64::decode(combinedRL);
                                auto decodedRR = Base64::decode(combinedRR);

                                if (decodedLL.empty() || decodedLR.empty() || decodedRL.empty() ||
                                    decodedRR.empty()) {
                                    nlohmann::json resp;
                                    resp["status"] = "error";
                                    resp["error_code"] = "IPC_INVALID_PARAMS";
                                    resp["message"] = "Failed to decode Base64 filter data";
                                    response = resp.dump();
                                } else {
                                    // TODO: Apply decoded filters to HRTFProcessor
                                    // This requires extending HRTFProcessor API to accept
                                    // pre-computed combined filters
                                    nlohmann::json resp;
                                    resp["status"] = "ok";
                                    resp["message"] = "Combined filter received";
                                    resp["data"]["rate_family"] = rateFamily;
                                    resp["data"]["ll_size"] = decodedLL.size();
                                    resp["data"]["lr_size"] = decodedLR.size();
                                    resp["data"]["rl_size"] = decodedRL.size();
                                    resp["data"]["rr_size"] = decodedRR.size();
                                    response = resp.dump();
                                    std::cout << "ZeroMQ: Received combined filter for "
                                              << rateFamily << " (LL:" << decodedLL.size()
                                              << " bytes)" << std::endl;
                                }
                            }
                        }
                    } else if (cmdType == "CROSSFEED_GET_STATUS") {
                        std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
                        bool enabled = g_crossfeed_enabled.load();
                        bool initialized = (g_hrtf_processor != nullptr);
                        nlohmann::json resp;
                        resp["status"] = "ok";
                        resp["data"]["enabled"] = enabled;
                        resp["data"]["initialized"] = initialized;
                        // TODO: Add head_size and headphone info when available
                        resp["data"]["head_size"] = "m";
                        resp["data"]["headphone"] = "";
                        response = resp.dump();
                    } else {
                        nlohmann::json resp;
                        resp["status"] = "error";
                        resp["error_code"] = "IPC_INVALID_COMMAND";
                        resp["message"] = "Unknown JSON command: " + cmdType;
                        response = resp.dump();
                    }
                } catch (const nlohmann::json::exception& e) {
                    nlohmann::json resp;
                    resp["status"] = "error";
                    resp["error_code"] = "IPC_PROTOCOL_ERROR";
                    resp["message"] = std::string("JSON parse error: ") + e.what();
                    response = resp.dump();
                }
            } else if (cmd == "PHASE_TYPE_GET") {
                // Get current phase type
                if (g_upsampler) {
                    PhaseType pt = g_upsampler->getPhaseType();
                    std::string ptStr = (pt == PhaseType::Minimum) ? "minimum" : "linear";
                    response = "OK:{\"phase_type\":\"" + ptStr + "\"}";
                } else {
                    response = "ERR:Upsampler not initialized";
                }
            } else if (cmd.rfind("PHASE_TYPE_SET:", 0) == 0) {
                // Set phase type: PHASE_TYPE_SET:minimum or PHASE_TYPE_SET:linear
                // Requires quad-phase mode to be enabled (4 filter variants preloaded)
                std::string phaseStr = cmd.substr(15);  // Extract after "PHASE_TYPE_SET:"
                if (!g_upsampler) {
                    response = "ERR:Upsampler not initialized";
                } else if (!g_upsampler->isQuadPhaseEnabled()) {
                    response = "ERR:Quad-phase mode not enabled (runtime switching unavailable)";
                } else if (phaseStr != "minimum" && phaseStr != "linear") {
                    response = "ERR:Invalid phase type (use 'minimum' or 'linear')";
                } else {
                    PhaseType newPhase =
                        (phaseStr == "minimum") ? PhaseType::Minimum : PhaseType::Linear;
                    PhaseType oldPhase = g_upsampler->getPhaseType();

                    if (oldPhase == newPhase) {
                        response = "OK:Phase type already " + phaseStr;
                    } else {
                        // Switch phase type (changes actual filter FFT in quad-phase mode)
                        if (!g_upsampler->switchPhaseType(newPhase)) {
                            response = "ERR:Failed to switch phase type";
                        } else {
                            // Re-apply EQ if enabled (switchPhaseType clears eqApplied_)
                            if (g_config.eqEnabled && !g_config.eqProfilePath.empty()) {
                                EQ::EqProfile eqProfile;
                                if (EQ::parseEqFile(g_config.eqProfilePath, eqProfile)) {
                                    size_t filterFftSize = g_upsampler->getFilterFftSize();
                                    size_t fullFftSize = g_upsampler->getFullFftSize();
                                    double outputSampleRate =
                                        static_cast<double>(g_config.inputSampleRate) *
                                        g_config.upsampleRatio;
                                    auto eqMagnitude = EQ::computeEqMagnitudeForFft(
                                        filterFftSize, fullFftSize, outputSampleRate, eqProfile);
                                    if (g_upsampler->applyEqMagnitude(eqMagnitude)) {
                                        std::cout << "ZeroMQ: EQ re-applied with " << phaseStr
                                                  << " phase" << std::endl;
                                    } else {
                                        std::cerr << "ZeroMQ: Warning - EQ re-apply failed"
                                                  << std::endl;
                                    }
                                } else {
                                    std::cerr << "ZeroMQ: Warning - Failed to parse EQ profile: "
                                              << g_config.eqProfilePath << std::endl;
                                }
                            }
                            response = "OK:Phase type set to " + phaseStr;
                        }
                    }
                }
            } else {
                response = "ERR:Unknown command";
            }

            // Send response
            zmq::message_t reply(response.data(), response.size());
            socket.send(reply, zmq::send_flags::none);
        }

        // Cleanup: unlink the IPC socket file (sock_path already defined at start)
        unlink(sock_path.c_str());
        std::cout << "ZeroMQ: Listener stopped" << std::endl;

    } catch (const zmq::error_t& e) {
        std::cerr << "ZeroMQ: Fatal error - " << e.what() << std::endl;
        std::cerr << "ZeroMQ: IPC communication disabled. Daemon will stop." << std::endl;
        g_zmq_bind_failed = true;
        g_running = false;
        // Quit main loop if running to trigger clean shutdown
        if (g_pw_loop) {
            pw_main_loop_quit(g_pw_loop);
        }
    }
}

// PipeWire objects
struct Data {
    struct pw_main_loop* loop;
    struct pw_stream* input_stream;
    bool gpu_ready;
};

// Signal handler for graceful shutdown
static void signal_handler(int sig) {
    if (sig == SIGHUP) {
        std::cout << "\nReceived SIGHUP, restarting for config reload..." << std::endl;
        g_reload_requested = true;
    } else {
        std::cout << "\nReceived signal " << sig << ", shutting down..." << std::endl;
    }

    // Start fade-out for glitch-free shutdown/restart (safe to call from signal handler)
    if (g_soft_mute) {
        g_soft_mute->startFadeOut();
    }

    // If main loop is running, quit it to trigger clean shutdown sequence
    // If not yet running (during startup/initialization), set g_running = false
    // to allow immediate abort
    if (g_main_loop_running.load() && g_pw_loop) {
        pw_main_loop_quit(g_pw_loop);
    } else {
        // Startup phase - allow immediate abort
        g_running = false;
    }
}

// Input stream process callback (44.1kHz audio from PipeWire)
static void on_input_process(void* userdata) {
    Data* data = static_cast<Data*>(userdata);

    struct pw_buffer* buf = pw_stream_dequeue_buffer(data->input_stream);
    if (!buf) {
        return;
    }

    struct spa_buffer* spa_buf = buf->buffer;
    float* input_samples = static_cast<float*>(spa_buf->datas[0].data);
    uint32_t n_frames = spa_buf->datas[0].chunk->size / (sizeof(float) * CHANNELS);

    if (input_samples && n_frames > 0 && data->gpu_ready) {
        // Deinterleave input (stereo interleaved → separate L/R)
        std::vector<float> left(n_frames);
        std::vector<float> right(n_frames);

        AudioUtils::deinterleaveStereo(input_samples, left.data(), right.data(), n_frames);

        // Process through GPU upsampler using streaming API
        // This preserves overlap buffer across calls
        std::vector<float> output_left, output_right;

        // Process left channel
        bool left_generated = g_upsampler->processStreamBlock(
            left.data(), n_frames, output_left, g_upsampler->streamLeft_, g_stream_input_left,
            g_stream_accumulated_left);

        // Process right channel
        bool right_generated = g_upsampler->processStreamBlock(
            right.data(), n_frames, output_right, g_upsampler->streamRight_, g_stream_input_right,
            g_stream_accumulated_right);

        // Only store output if both channels generated output
        // (they should synchronize due to same input size)
        if (left_generated && right_generated) {
            // Quick atomic check first to avoid lock overhead when crossfeed is disabled
            if (g_crossfeed_enabled.load()) {
                // Apply crossfeed (HRTF) - lock protects HRTFProcessor and CF buffers
                std::lock_guard<std::mutex> cf_lock(g_crossfeed_mutex);
                // Re-check under lock (double-checked locking pattern)
                if (g_hrtf_processor && g_hrtf_processor->isEnabled()) {
                    std::vector<float> cf_output_left, cf_output_right;
                    // Use default CUDA stream (0) for crossfeed processing
                    bool cf_generated = g_hrtf_processor->processStreamBlock(
                        output_left.data(), output_right.data(), output_left.size(), cf_output_left,
                        cf_output_right, 0, g_cf_stream_input_left, g_cf_stream_input_right,
                        g_cf_stream_accumulated_left, g_cf_stream_accumulated_right);

                    if (cf_generated) {
                        // Store crossfeed output for ALSA thread consumption
                        std::lock_guard<std::mutex> lock(g_buffer_mutex);
                        g_output_buffer_left.insert(g_output_buffer_left.end(),
                                                    cf_output_left.begin(), cf_output_left.end());
                        g_output_buffer_right.insert(g_output_buffer_right.end(),
                                                     cf_output_right.begin(),
                                                     cf_output_right.end());
                        g_buffer_cv.notify_one();
                    }
                    // Note: If crossfeed didn't generate output yet (accumulating),
                    // we don't store anything this iteration
                } else {
                    // Crossfeed was disabled while we were waiting for lock
                    std::lock_guard<std::mutex> lock(g_buffer_mutex);
                    g_output_buffer_left.insert(g_output_buffer_left.end(), output_left.begin(),
                                                output_left.end());
                    g_output_buffer_right.insert(g_output_buffer_right.end(), output_right.begin(),
                                                 output_right.end());
                    g_buffer_cv.notify_one();
                }
            } else {
                // Bypass crossfeed - store upsampler output directly
                std::lock_guard<std::mutex> lock(g_buffer_mutex);
                g_output_buffer_left.insert(g_output_buffer_left.end(), output_left.begin(),
                                            output_left.end());
                g_output_buffer_right.insert(g_output_buffer_right.end(), output_right.begin(),
                                             output_right.end());
                g_buffer_cv.notify_one();
            }
        }
    }

    pw_stream_queue_buffer(data->input_stream, buf);
}

// Stream state changed callback
static void on_stream_state_changed(void* userdata, enum pw_stream_state old_state,
                                    enum pw_stream_state state, const char* error) {
    (void)userdata;
    (void)old_state;

    std::cout << "PipeWire input stream state: " << pw_stream_state_as_string(state);
    if (error) {
        std::cout << " (error: " << error << ")";
    }
    std::cout << std::endl;
}

static const struct pw_stream_events input_stream_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .state_changed = on_stream_state_changed,
    .process = on_input_process,
};

// Open and configure ALSA device. Returns nullptr on failure.
static snd_pcm_t* open_and_configure_pcm() {
    snd_pcm_t* pcm_handle = nullptr;
    int err;

    err = snd_pcm_open(&pcm_handle, g_config.alsaDevice.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) {
        std::cerr << "ALSA: Cannot open device " << g_config.alsaDevice << ": " << snd_strerror(err)
                  << std::endl;
        return nullptr;
    }

    // Set hardware parameters
    snd_pcm_hw_params_t* hw_params;
    snd_pcm_hw_params_alloca(&hw_params);
    snd_pcm_hw_params_any(pcm_handle, hw_params);

    if ((err = snd_pcm_hw_params_set_access(pcm_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) <
            0 ||
        (err = snd_pcm_hw_params_set_format(pcm_handle, hw_params, SND_PCM_FORMAT_S32_LE)) < 0) {
        std::cerr << "ALSA: Cannot set access/format: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return nullptr;
    }

    // Calculate output sample rate from config
    unsigned int rate = g_config.inputSampleRate * g_config.upsampleRatio;
    if ((err = snd_pcm_hw_params_set_rate_near(pcm_handle, hw_params, &rate, 0)) < 0) {
        std::cerr << "ALSA: Cannot set sample rate: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return nullptr;
    }
    if ((err = snd_pcm_hw_params_set_channels(pcm_handle, hw_params, CHANNELS)) < 0) {
        std::cerr << "ALSA: Cannot set channel count: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return nullptr;
    }

    snd_pcm_uframes_t buffer_size = static_cast<snd_pcm_uframes_t>(g_config.bufferSize);
    snd_pcm_uframes_t period_size = static_cast<snd_pcm_uframes_t>(g_config.periodSize);
    snd_pcm_hw_params_set_buffer_size_near(pcm_handle, hw_params, &buffer_size);
    snd_pcm_hw_params_set_period_size_near(pcm_handle, hw_params, &period_size, 0);

    if ((err = snd_pcm_hw_params(pcm_handle, hw_params)) < 0) {
        std::cerr << "ALSA: Cannot set hardware parameters: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return nullptr;
    }

    if ((err = snd_pcm_prepare(pcm_handle)) < 0) {
        std::cerr << "ALSA: Cannot prepare device: " << snd_strerror(err) << std::endl;
        snd_pcm_close(pcm_handle);
        return nullptr;
    }

    std::cout << "ALSA: Output device configured (" << rate << " Hz, 32-bit int, stereo)"
              << " buffer " << buffer_size << " frames, period " << period_size << " frames"
              << std::endl;
    return pcm_handle;
}

// Check current PCM state; return false if disconnected/suspended
static bool pcm_alive(snd_pcm_t* pcm_handle) {
    if (!pcm_handle)
        return false;
    snd_pcm_status_t* status;
    snd_pcm_status_alloca(&status);
    if (snd_pcm_status(pcm_handle, status) < 0) {
        return false;
    }
    snd_pcm_state_t st = snd_pcm_status_get_state(status);
    if (st == SND_PCM_STATE_DISCONNECTED || st == SND_PCM_STATE_SUSPENDED) {
        return false;
    }
    return true;
}

// ALSA output thread (705.6kHz direct to DAC)
void alsa_output_thread() {
    snd_pcm_t* pcm_handle = open_and_configure_pcm();
    std::vector<int32_t> interleaved_buffer(32768 * CHANNELS);  // resized after open
    std::vector<float> float_buffer(32768 * CHANNELS);          // for soft mute processing
    snd_pcm_uframes_t period_size = 32768;
    if (!pcm_handle) {
        // Retry loop until device appears or shutdown
        while (g_running && !pcm_handle) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            pcm_handle = open_and_configure_pcm();
        }
    }
    if (!pcm_handle) {
        return;
    }
    snd_pcm_hw_params_t* cur_params;
    snd_pcm_hw_params_alloca(&cur_params);
    if (snd_pcm_hw_params_current(pcm_handle, cur_params) == 0) {
        snd_pcm_uframes_t detected_period = 0;
        if (snd_pcm_hw_params_get_period_size(cur_params, &detected_period, nullptr) == 0 &&
            detected_period > 0) {
            period_size = detected_period;
            interleaved_buffer.resize(period_size * CHANNELS);
            float_buffer.resize(period_size * CHANNELS);
        }
    }

    // Main playback loop
    while (g_running) {
        // Heartbeat check every few hundred loops
        static int alive_counter = 0;
        if (++alive_counter > 200) {  // ~200 iterations ~ a few seconds depending on buffer wait
            alive_counter = 0;
            if (!pcm_alive(pcm_handle)) {
                std::cerr << "ALSA: PCM disconnected/suspended, attempting reopen..." << std::endl;
                if (pcm_handle) {
                    snd_pcm_close(pcm_handle);
                    pcm_handle = nullptr;
                }
                while (g_running && !pcm_handle) {
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    pcm_handle = open_and_configure_pcm();
                }
                // Reset buffer positions to avoid backlog after long downtime
                {
                    std::lock_guard<std::mutex> lock(g_buffer_mutex);
                    g_output_buffer_left.clear();
                    g_output_buffer_right.clear();
                    g_output_read_pos = 0;
                }
                continue;
            }
        }

        // Wait for GPU processed data (3x period to ensure sufficient buffering)
        std::unique_lock<std::mutex> lock(g_buffer_mutex);
        g_buffer_cv.wait_for(lock, std::chrono::milliseconds(200), [period_size] {
            return (g_output_buffer_left.size() - g_output_read_pos) >= (period_size * 3) ||
                   !g_running;
        });

        if (!g_running)
            break;

        size_t available = g_output_buffer_left.size() - g_output_read_pos;
        if (available >= period_size) {
            // Step 1: Interleave L/R channels into float buffer with gain applied
            const float gain = g_config.gain;
            AudioUtils::interleaveStereoWithGain(g_output_buffer_left.data() + g_output_read_pos,
                                                 g_output_buffer_right.data() + g_output_read_pos,
                                                 float_buffer.data(), period_size, gain);

            // Step 2: Apply soft mute for glitch-free shutdown/transitions
            if (g_soft_mute) {
                g_soft_mute->process(float_buffer.data(), period_size);
            }

            // Step 3: Clipping detection, clamping, and float→int32 conversion
            static auto last_stats_write = std::chrono::steady_clock::now();
            size_t current_clips = 0;

            for (size_t i = 0; i < period_size; ++i) {
                float left_sample = float_buffer[i * 2];
                float right_sample = float_buffer[i * 2 + 1];

                // Detect and count clipping (for diagnostics only)
                if (left_sample > 1.0f || left_sample < -1.0f || right_sample > 1.0f ||
                    right_sample < -1.0f) {
                    current_clips++;
                    g_clip_count.fetch_add(1, std::memory_order_relaxed);
                }

                // Hard clipping as safety net only
                left_sample = std::clamp(left_sample, -1.0f, 1.0f);
                right_sample = std::clamp(right_sample, -1.0f, 1.0f);

                // Float to int32 conversion
                constexpr float INT32_MAX_FLOAT = 2147483647.0f;
                interleaved_buffer[i * 2] =
                    static_cast<int32_t>(std::lroundf(left_sample * INT32_MAX_FLOAT));
                interleaved_buffer[i * 2 + 1] =
                    static_cast<int32_t>(std::lroundf(right_sample * INT32_MAX_FLOAT));
            }

            g_total_samples.fetch_add(period_size * 2, std::memory_order_relaxed);

            // Report clipping infrequently to avoid log spam
            size_t total = g_total_samples.load(std::memory_order_relaxed);
            size_t clips = g_clip_count.load(std::memory_order_relaxed);
            if (total % (period_size * 2 * 100) == 0 && clips > 0) {
                std::cout << "WARNING: Clipping detected - " << clips << " samples clipped out of "
                          << total << " (" << (100.0 * clips / total) << "%)" << std::endl;
            }

            // Write stats file every second
            auto now = std::chrono::steady_clock::now();
            if (now - last_stats_write >= std::chrono::seconds(1)) {
                write_stats_file();
                last_stats_write = now;
            }
            g_output_read_pos += period_size;

            // Clear consumed data more frequently to avoid large vector reallocations
            // Reallocations can cause audio glitches due to memory operations
            if (g_output_read_pos > period_size * 4) {  // Clean up after 4 periods
                g_output_buffer_left.erase(g_output_buffer_left.begin(),
                                           g_output_buffer_left.begin() + g_output_read_pos);
                g_output_buffer_right.erase(g_output_buffer_right.begin(),
                                            g_output_buffer_right.begin() + g_output_read_pos);
                g_output_read_pos = 0;
            }

            lock.unlock();

            // Write to ALSA device
            snd_pcm_sframes_t frames_written =
                snd_pcm_writei(pcm_handle, interleaved_buffer.data(), period_size);
            if (frames_written < 0) {
                // Try standard recovery first
                snd_pcm_sframes_t rec = snd_pcm_recover(pcm_handle, frames_written, 0);
                if (rec < 0) {
                    // Device may be gone; attempt reopen
                    std::cerr << "ALSA: Write error: " << snd_strerror(frames_written)
                              << " (recover=" << snd_strerror(rec) << "), retrying reopen..."
                              << std::endl;
                    snd_pcm_close(pcm_handle);
                    pcm_handle = nullptr;
                    while (g_running && !pcm_handle) {
                        std::this_thread::sleep_for(std::chrono::seconds(5));
                        pcm_handle = open_and_configure_pcm();
                    }
                    if (pcm_handle) {
                        // resize buffer to new period size if needed
                        snd_pcm_uframes_t new_period = 0;
                        snd_pcm_hw_params_t* hw_params;
                        snd_pcm_hw_params_alloca(&hw_params);
                        if (snd_pcm_hw_params_current(pcm_handle, hw_params) == 0 &&
                            snd_pcm_hw_params_get_period_size(hw_params, &new_period, nullptr) ==
                                0 &&
                            new_period != period_size) {
                            period_size = new_period;
                            interleaved_buffer.resize(period_size * CHANNELS);
                            float_buffer.resize(period_size * CHANNELS);
                        }
                        // Drop queued buffers on successful reopen to avoid burst
                        {
                            std::lock_guard<std::mutex> lock(g_buffer_mutex);
                            g_output_buffer_left.clear();
                            g_output_buffer_right.clear();
                            g_output_read_pos = 0;
                        }
                    } else {
                        // If still not available, continue loop to retry
                        continue;
                    }
                }
            }
        } else {
            // Not enough data → write silence to keep device fed and avoid xrun
            lock.unlock();

            // Apply soft mute even to silence (to advance fade position during shutdown)
            if (g_soft_mute && g_soft_mute->isTransitioning()) {
                std::fill(float_buffer.begin(), float_buffer.end(), 0.0f);
                g_soft_mute->process(float_buffer.data(), period_size);
            }

            std::fill(interleaved_buffer.begin(), interleaved_buffer.end(), 0);
            snd_pcm_sframes_t frames_written =
                snd_pcm_writei(pcm_handle, interleaved_buffer.data(), period_size);
            if (frames_written < 0) {
                snd_pcm_sframes_t rec = snd_pcm_recover(pcm_handle, frames_written, 0);
                if (rec < 0) {
                    std::cerr << "ALSA: Silence write error: " << snd_strerror(frames_written)
                              << " (recover=" << snd_strerror(rec) << "), retrying reopen..."
                              << std::endl;
                    if (pcm_handle) {
                        snd_pcm_close(pcm_handle);
                        pcm_handle = nullptr;
                    }
                    while (g_running && !pcm_handle) {
                        std::this_thread::sleep_for(std::chrono::seconds(5));
                        pcm_handle = open_and_configure_pcm();
                    }
                }
            }
        }
    }

    // Cleanup
    if (pcm_handle) {
        snd_pcm_drain(pcm_handle);
        snd_pcm_close(pcm_handle);
    }
    std::cout << "ALSA: Output thread terminated" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  GPU Audio Upsampler - ALSA Direct Output" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Install signal handlers (SIGHUP for restart, SIGINT/SIGTERM for shutdown)
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGHUP, signal_handler);

    // Acquire PID file lock (prevent multiple instances)
    if (!acquire_pid_lock()) {
        return 1;
    }
    std::cout << "PID: " << getpid() << " (file: " << PID_FILE_PATH << ")" << std::endl;

    int exitCode = 0;

    do {
        g_running = true;
        g_reload_requested = false;
        reset_runtime_state();

        // Load configuration from config.json (if exists)
        load_runtime_config();

        // Environment variable overrides config.json
        if (const char* env_dev = std::getenv("ALSA_DEVICE")) {
            g_config.alsaDevice = env_dev;
            std::cout << "Config: ALSA_DEVICE env override: " << env_dev << std::endl;
        }

        // Command line argument overrides filter path
        if (argc > 1) {
            g_config.filterPath = argv[1];
            std::cout << "Config: CLI filter path override: " << argv[1] << std::endl;
        }

        // Auto-select filter based on sample rate if configured filter doesn't exist
        // but a sample-rate-specific version does
        if (!std::filesystem::exists(g_config.filterPath)) {
            // Try to find sample-rate-specific filter
            std::string basePath = g_config.filterPath;
            size_t dotPos = basePath.rfind('.');
            if (dotPos != std::string::npos) {
                std::string rateSpecificPath = basePath.substr(0, dotPos) + "_" +
                                               std::to_string(g_config.inputSampleRate) +
                                               basePath.substr(dotPos);
                if (std::filesystem::exists(rateSpecificPath)) {
                    std::cout << "Config: Using sample-rate-specific filter: " << rateSpecificPath
                              << std::endl;
                    g_config.filterPath = rateSpecificPath;
                }
            }
        }

        // Warn if using 44.1kHz filter with 48kHz input
        if (g_config.inputSampleRate == 48000 &&
            g_config.filterPath.find("44100") == std::string::npos &&
            g_config.filterPath.find("48000") == std::string::npos) {
            std::cout << "Warning: Using generic filter with 48kHz input. "
                      << "For optimal quality, generate a 48kHz-optimized filter." << std::endl;
        }

        // Initialize GPU upsampler with configured values
        std::cout << "Initializing GPU upsampler..." << std::endl;
        g_upsampler = new ConvolutionEngine::GPUUpsampler();

        bool initSuccess = false;
        if (g_config.quadPhaseEnabled) {
            // Quad-phase mode: load all 4 filter configurations
            std::cout << "Quad-phase mode enabled" << std::endl;

            // Check all 4 filter files exist
            bool allFilesExist = true;
            for (const auto& path : {g_config.filterPath44kMin, g_config.filterPath48kMin,
                                     g_config.filterPath44kLinear, g_config.filterPath48kLinear}) {
                if (!std::filesystem::exists(path)) {
                    std::cerr << "Config error: Filter file not found: " << path << std::endl;
                    allFilesExist = false;
                }
            }
            if (!allFilesExist) {
                delete g_upsampler;
                exitCode = 1;
                break;
            }

            // Determine initial rate family from inputSampleRate
            ConvolutionEngine::RateFamily initialFamily =
                ConvolutionEngine::detectRateFamily(g_config.inputSampleRate);
            if (initialFamily == ConvolutionEngine::RateFamily::RATE_UNKNOWN) {
                initialFamily = ConvolutionEngine::RateFamily::RATE_44K;
            }

            initSuccess = g_upsampler->initializeQuadPhase(
                g_config.filterPath44kMin, g_config.filterPath48kMin, g_config.filterPath44kLinear,
                g_config.filterPath48kLinear, g_config.upsampleRatio, g_config.blockSize,
                initialFamily, g_config.phaseType);
        } else {
            // Standard single-filter mode (backward compatible)
            if (!std::filesystem::exists(g_config.filterPath)) {
                std::cerr << "Config error: Filter file not found: " << g_config.filterPath
                          << std::endl;
                delete g_upsampler;
                exitCode = 1;
                break;
            }
            initSuccess = g_upsampler->initialize(g_config.filterPath, g_config.upsampleRatio,
                                                  g_config.blockSize);
        }

        if (!initSuccess) {
            std::cerr << "Failed to initialize GPU upsampler" << std::endl;
            delete g_upsampler;
            exitCode = 1;
            break;
        }

        // Check for early abort (signal received during GPU initialization)
        if (!g_running) {
            std::cout << "Startup interrupted by signal" << std::endl;
            delete g_upsampler;
            g_upsampler = nullptr;
            break;
        }

        std::cout << "GPU upsampler ready (" << g_config.upsampleRatio << "x upsampling, "
                  << g_config.blockSize << " samples/block)" << std::endl;

        // Set input sample rate for correct output rate calculation (non-quad-phase mode)
        if (!g_config.quadPhaseEnabled) {
            g_upsampler->setInputSampleRate(g_config.inputSampleRate);
            g_upsampler->setPhaseType(g_config.phaseType);
        }
        std::cout << "Input sample rate: " << g_config.inputSampleRate << " Hz -> "
                  << g_upsampler->getOutputSampleRate() << " Hz output" << std::endl;
        std::cout << "Phase type: " << phaseTypeToString(g_config.phaseType) << std::endl;

        // Log latency warning for linear phase
        if (g_config.phaseType == PhaseType::Linear) {
            double latencySec = g_upsampler->getLatencySeconds();
            std::cout << "  WARNING: Linear phase latency: " << latencySec << " seconds ("
                      << g_upsampler->getLatencySamples() << " samples)" << std::endl;
        }

        // Initialize streaming mode to preserve overlap buffers across PipeWire callbacks
        if (!g_upsampler->initializeStreaming()) {
            std::cerr << "Failed to initialize streaming mode" << std::endl;
            delete g_upsampler;
            exitCode = 1;
            break;
        }

        // Check for early abort
        if (!g_running) {
            std::cout << "Startup interrupted by signal" << std::endl;
            delete g_upsampler;
            g_upsampler = nullptr;
            break;
        }

        // Apply EQ profile if enabled
        if (g_config.eqEnabled && !g_config.eqProfilePath.empty()) {
            std::cout << "Loading EQ profile: " << g_config.eqProfilePath << std::endl;
            EQ::EqProfile eqProfile;
            if (EQ::parseEqFile(g_config.eqProfilePath, eqProfile)) {
                std::cout << "  EQ: " << eqProfile.name << " (" << eqProfile.bands.size()
                          << " bands, preamp " << eqProfile.preampDb << " dB)" << std::endl;

                // Compute EQ magnitude response and apply with minimum phase reconstruction
                size_t filterFftSize = g_upsampler->getFilterFftSize();  // N/2+1 (R2C output)
                size_t fullFftSize = g_upsampler->getFullFftSize();      // N (full FFT)
                double outputSampleRate =
                    static_cast<double>(g_config.inputSampleRate) * g_config.upsampleRatio;
                auto eqMagnitude = EQ::computeEqMagnitudeForFft(filterFftSize, fullFftSize,
                                                                outputSampleRate, eqProfile);

                if (g_upsampler->applyEqMagnitude(eqMagnitude)) {
                    // Log message depends on phase type (already logged by applyEqMagnitude)
                } else {
                    std::cerr << "  EQ: Failed to apply frequency response" << std::endl;
                }
            } else {
                std::cerr << "  EQ: Failed to parse profile: " << g_config.eqProfilePath
                          << std::endl;
            }
        }

        // Pre-allocate streaming input buffers (based on streamValidInputPerBlock_)
        // Use 2x safety margin to handle timing variations
        size_t buffer_capacity = g_upsampler->getStreamValidInputPerBlock() * 2;
        g_stream_input_left.resize(buffer_capacity, 0.0f);
        g_stream_input_right.resize(buffer_capacity, 0.0f);
        std::cout << "Streaming buffer capacity: " << buffer_capacity
                  << " samples (2x streamValidInputPerBlock)" << std::endl;
        g_stream_accumulated_left = 0;
        g_stream_accumulated_right = 0;

        // Initialize HRTF processor for crossfeed (optional feature)
        // Crossfeed is disabled by default until enabled via ZeroMQ command
        std::string hrtfDir = "data/crossfeed/hrtf";
        if (std::filesystem::exists(hrtfDir)) {
            std::cout << "Initializing HRTF processor for crossfeed..." << std::endl;
            g_hrtf_processor = new CrossfeedEngine::HRTFProcessor();

            // Determine rate family based on input sample rate
            CrossfeedEngine::RateFamily rateFamily = (g_config.inputSampleRate == 48000)
                                                         ? CrossfeedEngine::RateFamily::RATE_48K
                                                         : CrossfeedEngine::RateFamily::RATE_44K;

            if (g_hrtf_processor->initialize(hrtfDir, g_config.blockSize,
                                             CrossfeedEngine::HeadSize::M, rateFamily)) {
                if (g_hrtf_processor->initializeStreaming()) {
                    std::cout << "  HRTF processor ready (head size: M, rate family: "
                              << (rateFamily == CrossfeedEngine::RateFamily::RATE_44K ? "44k"
                                                                                      : "48k")
                              << ")" << std::endl;

                    // Pre-allocate crossfeed streaming buffers
                    size_t cf_buffer_capacity = g_hrtf_processor->getStreamValidInputPerBlock() * 2;
                    g_cf_stream_input_left.resize(cf_buffer_capacity, 0.0f);
                    g_cf_stream_input_right.resize(cf_buffer_capacity, 0.0f);
                    g_cf_stream_accumulated_left = 0;
                    g_cf_stream_accumulated_right = 0;
                    std::cout << "  Crossfeed buffer capacity: " << cf_buffer_capacity << " samples"
                              << std::endl;

                    // Crossfeed is initialized but disabled by default
                    g_crossfeed_enabled.store(false);
                    g_hrtf_processor->setEnabled(false);
                    std::cout << "  Crossfeed: initialized (disabled by default)" << std::endl;
                } else {
                    std::cerr << "  HRTF: Failed to initialize streaming mode" << std::endl;
                    delete g_hrtf_processor;
                    g_hrtf_processor = nullptr;
                }
            } else {
                std::cerr << "  HRTF: Failed to initialize processor" << std::endl;
                std::cerr
                    << "  Hint: Run 'uv run python scripts/generate_hrtf.py' to generate HRTF "
                       "filters"
                    << std::endl;
                delete g_hrtf_processor;
                g_hrtf_processor = nullptr;
            }
        } else {
            std::cout << "HRTF directory not found (" << hrtfDir << "), crossfeed feature disabled"
                      << std::endl;
            std::cout
                << "  Hint: Run 'uv run python scripts/generate_hrtf.py' to generate HRTF filters"
                << std::endl;
        }

        std::cout << std::endl;

        // Check for early abort before starting threads
        if (!g_running) {
            std::cout << "Startup interrupted by signal" << std::endl;
            delete g_upsampler;
            g_upsampler = nullptr;
            break;
        }

        // Initialize soft mute controller with output sample rate (50ms fade duration)
        int outputSampleRate = g_config.inputSampleRate * g_config.upsampleRatio;
        g_soft_mute = new SoftMute::Controller(50, outputSampleRate);
        std::cout << "Soft mute initialized (50ms fade at " << outputSampleRate << "Hz)"
                  << std::endl;

        // Start ALSA output thread
        std::cout << "Starting ALSA output thread..." << std::endl;
        std::thread alsa_thread(alsa_output_thread);

        // Initialize PipeWire input
        pw_init(&argc, &argv);

        Data data = {};
        data.gpu_ready = true;

        // Create main loop (store globally for signal handler access)
        data.loop = pw_main_loop_new(nullptr);
        g_pw_loop = data.loop;

        // Start ZeroMQ listener thread AFTER g_pw_loop is set
        // This ensures RELOAD commands can always quit the main loop
        std::thread zmq_thread(zeromq_listener_thread);
        struct pw_loop* loop = pw_main_loop_get_loop(data.loop);

        // Create Capture stream from gpu_upsampler_sink.monitor
        std::cout << "Creating PipeWire input (capturing from gpu_upsampler_sink)..." << std::endl;
        data.input_stream = pw_stream_new_simple(
            loop, "GPU Upsampler Input",
            pw_properties_new(PW_KEY_MEDIA_TYPE, "Audio", PW_KEY_MEDIA_CATEGORY, "Capture",
                              PW_KEY_MEDIA_ROLE, "Music", PW_KEY_NODE_DESCRIPTION,
                              "GPU Upsampler Input", PW_KEY_NODE_TARGET,
                              "gpu_upsampler_sink.monitor", "audio.channels", "2", "audio.position",
                              "FL,FR", nullptr),
            &input_stream_events, &data);

        // Configure input stream audio format (32-bit float stereo)
        uint8_t input_buffer[1024];
        struct spa_pod_builder input_builder =
            SPA_POD_BUILDER_INIT(input_buffer, sizeof(input_buffer));

        struct spa_audio_info_raw input_info = {};
        input_info.format = SPA_AUDIO_FORMAT_F32;
        input_info.rate = g_config.inputSampleRate;
        input_info.channels = CHANNELS;
        input_info.position[0] = SPA_AUDIO_CHANNEL_FL;
        input_info.position[1] = SPA_AUDIO_CHANNEL_FR;

        const struct spa_pod* input_params[1];
        input_params[0] =
            spa_format_audio_raw_build(&input_builder, SPA_PARAM_EnumFormat, &input_info);

        pw_stream_connect(
            data.input_stream, PW_DIRECTION_INPUT, PW_ID_ANY,
            static_cast<pw_stream_flags>(PW_STREAM_FLAG_MAP_BUFFERS | PW_STREAM_FLAG_RT_PROCESS),
            input_params, 1);

        double outputRateKHz = g_config.inputSampleRate * g_config.upsampleRatio / 1000.0;
        std::cout << std::endl;
        std::cout << "System ready. Audio routing configured:" << std::endl;
        std::cout << "  1. Applications → gpu_upsampler_sink (select in GNOME settings)"
                  << std::endl;
        std::cout << "  2. gpu_upsampler_sink.monitor → GPU Upsampler (" << g_config.upsampleRatio
                  << "x upsampling)" << std::endl;
        std::cout << "  3. GPU Upsampler → ALSA → SMSL DAC (" << outputRateKHz << "kHz direct)"
                  << std::endl;
        std::cout << std::endl;
        std::cout << "Select 'GPU Upsampler (" << outputRateKHz
                  << "kHz)' as output device in sound settings." << std::endl;
        std::cout << "Press Ctrl+C to stop." << std::endl;
        std::cout << "========================================" << std::endl;

        // Check if RELOAD was requested during startup (before main loop),
        // ZeroMQ bind failed, or signal received during initialization
        if (g_running.load() && !g_reload_requested.load() && !g_zmq_bind_failed.load()) {
            // Run main loop (only if not already requested to reload/stop)
            g_main_loop_running = true;
            pw_main_loop_run(data.loop);
            g_main_loop_running = false;
        } else if (g_zmq_bind_failed.load()) {
            std::cerr << "Startup aborted due to ZeroMQ bind failure." << std::endl;
        } else if (!g_running.load()) {
            std::cout << "Startup interrupted by signal, skipping main loop." << std::endl;
        } else {
            std::cout << "RELOAD requested during startup, skipping main loop." << std::endl;
        }

        // Cleanup
        std::cout << "Shutting down..." << std::endl;

        // Wait for soft mute fade-out to complete (with 100ms timeout)
        // This ensures glitch-free audio shutdown
        if (g_soft_mute && g_soft_mute->isTransitioning()) {
            std::cout << "Waiting for fade-out to complete..." << std::endl;
            auto fade_start = std::chrono::steady_clock::now();
            while (g_soft_mute->isTransitioning()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                if (std::chrono::steady_clock::now() - fade_start >
                    std::chrono::milliseconds(100)) {
                    std::cout << "Fade-out timeout, forcing shutdown" << std::endl;
                    break;
                }
            }
        }

        if (data.input_stream) {
            pw_stream_destroy(data.input_stream);
        }
        if (data.loop) {
            pw_main_loop_destroy(data.loop);
            g_pw_loop = nullptr;
        }

        g_running = false;
        g_buffer_cv.notify_all();
        zmq_thread.join();
        alsa_thread.join();

        delete g_soft_mute;
        g_soft_mute = nullptr;
        delete g_hrtf_processor;
        g_hrtf_processor = nullptr;
        g_crossfeed_enabled.store(false);
        delete g_upsampler;
        g_upsampler = nullptr;
        pw_deinit();

        // Don't reload if ZMQ bind failed - exit completely
        if (g_zmq_bind_failed) {
            std::cerr << "Exiting due to ZeroMQ initialization failure." << std::endl;
            exitCode = 1;
            break;
        }

        if (g_reload_requested) {
            std::cout << "Reload requested. Restarting daemon with updated config..." << std::endl;
        }
    } while (g_reload_requested);

    // Release PID lock and remove file on clean exit
    release_pid_lock();
    std::cout << "Goodbye!" << std::endl;
    return exitCode;
}
