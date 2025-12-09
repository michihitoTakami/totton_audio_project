(function () {
    const DEFAULT_TRANSLATIONS = {
        connected: "Connected",
        disconnected: "Disconnected",
        listening: "Listening",
        stopped: "Stopped",
        noClient: "No client",
        unknown: "—",
        startSuccess: "TCP input started",
        stopSuccess: "TCP input stopped",
        updateSuccess: "Config updated",
        updateError: "Failed to update config",
        startError: "Failed to start TCP input",
        stopError: "Failed to stop TCP input",
    };

    function defaultSettings() {
        return {
            enabled: true,
            bind_address: "0.0.0.0",
            port: 46001,
            buffer_size_bytes: 262144,
            connection_mode: "single",
            priority_clients: [],
        };
    }

    function defaultTelemetry() {
        return {
            listening: false,
            bound_port: null,
            client_connected: false,
            streaming: false,
            client_address: null,
            uptime_seconds: null,
            xrun_count: 0,
            ring_buffer_frames: 0,
            watermark_frames: 0,
            buffered_frames: 0,
            max_buffered_frames: 0,
            dropped_frames: 0,
            disconnect_reason: null,
            connection_mode: "single",
            priority_clients: [],
            last_header: null,
            rep_endpoint: null,
            pub_endpoint: null,
        };
    }

    function formatNumber(value) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return "—";
        }
        return Number(value).toLocaleString();
    }

    function formatBytes(bytes) {
        if (bytes === null || bytes === undefined || Number.isNaN(Number(bytes))) {
            return "—";
        }
        const num = Number(bytes);
        if (num < 1024) return `${num.toFixed(0)} B`;
        if (num < 1024 * 1024) return `${(num / 1024).toFixed(1)} KB`;
        if (num < 1024 * 1024 * 1024) return `${(num / 1024 / 1024).toFixed(1)} MB`;
        return `${(num / 1024 / 1024 / 1024).toFixed(1)} GB`;
    }

    function formatDuration(seconds) {
        if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) {
            return "—";
        }
        const total = Math.max(0, Math.floor(Number(seconds)));
        const hours = Math.floor(total / 3600);
        const minutes = Math.floor((total % 3600) / 60);
        const secs = total % 60;
        const parts = [];
        if (hours > 0) parts.push(hours.toString().padStart(2, "0"));
        parts.push(minutes.toString().padStart(2, "0"));
        parts.push(secs.toString().padStart(2, "0"));
        return parts.join(":");
    }

    function formatTimestamp(ts) {
        if (!ts) return "—";
        const date = typeof ts === "number" ? new Date(ts) : new Date(String(ts));
        if (Number.isNaN(date.getTime())) return "—";
        return date.toLocaleString();
    }

    function bitDepthFromFormat(fmt) {
        if (!fmt) return null;
        const format = String(fmt).toUpperCase();
        if (format.includes("16")) return 16;
        if (format.includes("24")) return 24;
        if (format.includes("32")) return 32;
        return null;
    }

    function bytesPerSampleFromFormat(fmt) {
        const depth = bitDepthFromFormat(fmt);
        return depth ? depth / 8 : null;
    }

    function createTcpInputPage(options = {}) {
        const pollingMs = options.pollingMs || 2000;
        const translations = Object.assign({}, DEFAULT_TRANSLATIONS, options.translations || {});

        return {
            translations,
            formatNumber,
            formatBytes,
            formatDuration,
            formatTimestamp,
            status: {
                settings: defaultSettings(),
                telemetry: defaultTelemetry(),
                updatedAt: null,
            },
            form: {
                bindAddress: "0.0.0.0",
                port: 46001,
                bufferSizeBytes: 262144,
                connectionMode: "single",
                priorityClientsText: "",
            },
            actionInProgress: false,
            errorMessage: "",

            init() {
                this.fetchStatus();
                this._interval = setInterval(() => this.fetchStatus(), pollingMs);
            },

            async fetchStatus() {
                try {
                    const response = await fetch("/api/tcp-input/status");
                    if (!response.ok) {
                        throw new Error("Failed to fetch TCP input status");
                    }
                    const data = await response.json();
                    if (data.settings) {
                        this.status.settings = data.settings;
                        this.syncFormFromSettings(data.settings);
                    }
                    if (data.telemetry) {
                        this.status.telemetry = data.telemetry;
                    }
                    this.status.updatedAt = Date.now();
                    this.errorMessage = "";
                } catch (error) {
                    console.error("Failed to load TCP input status", error);
                    this.errorMessage = error.message || "Status fetch failed";
                }
            },

            syncFormFromSettings(settings) {
                this.form.bindAddress = settings.bind_address ?? this.form.bindAddress;
                this.form.port = settings.port ?? this.form.port;
                this.form.bufferSizeBytes =
                    settings.buffer_size_bytes ?? this.form.bufferSizeBytes;
                this.form.connectionMode = settings.connection_mode ?? this.form.connectionMode;
                this.form.priorityClientsText = Array.isArray(settings.priority_clients)
                    ? settings.priority_clients.join("\n")
                    : this.form.priorityClientsText;
            },

            priorityClientsList() {
                return (this.form.priorityClientsText || "")
                    .split(/\r?\n/)
                    .map((c) => c.trim())
                    .filter(Boolean);
            },

            async startServer() {
                if (this.actionInProgress) return;
                this.actionInProgress = true;
                try {
                    const response = await fetch("/api/tcp-input/start", { method: "POST" });
                    const data = await response.json();
                    if (!response.ok || data.success === false) {
                        throw new Error(data.detail?.message || data.message || "start failed");
                    }
                    showToast(data.message || this.translations.startSuccess, "success");
                    await this.fetchStatus();
                } catch (error) {
                    console.error("Failed to start TCP input", error);
                    showToast(this.translations.startError, "error");
                    this.errorMessage = error.message || this.translations.startError;
                } finally {
                    this.actionInProgress = false;
                }
            },

            async stopServer() {
                if (this.actionInProgress) return;
                this.actionInProgress = true;
                try {
                    const response = await fetch("/api/tcp-input/stop", { method: "POST" });
                    const data = await response.json();
                    if (!response.ok || data.success === false) {
                        throw new Error(data.detail?.message || data.message || "stop failed");
                    }
                    showToast(data.message || this.translations.stopSuccess, "success");
                    await this.fetchStatus();
                } catch (error) {
                    console.error("Failed to stop TCP input", error);
                    showToast(this.translations.stopError, "error");
                    this.errorMessage = error.message || this.translations.stopError;
                } finally {
                    this.actionInProgress = false;
                }
            },

            async updateConfig() {
                if (this.actionInProgress) return;
                this.actionInProgress = true;
                try {
                    const payload = {
                        bind_address: this.form.bindAddress,
                        port: Number(this.form.port),
                        buffer_size_bytes: Number(this.form.bufferSizeBytes),
                        connection_mode: this.form.connectionMode,
                        priority_clients: this.priorityClientsList(),
                    };
                    const response = await fetch("/api/tcp-input/config", {
                        method: "PUT",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload),
                    });
                    const data = await response.json();
                    if (!response.ok || data.success === false) {
                        throw new Error(data.detail?.message || data.message || "update failed");
                    }
                    showToast(data.message || this.translations.updateSuccess, "success");
                    await this.fetchStatus();
                } catch (error) {
                    console.error("Failed to update TCP config", error);
                    showToast(this.translations.updateError, "error");
                    this.errorMessage = error.message || this.translations.updateError;
                } finally {
                    this.actionInProgress = false;
                }
            },

            connectionText() {
                if (this.status.telemetry.listening) {
                    return this.status.telemetry.client_connected
                        ? this.translations.connected
                        : this.translations.listening;
                }
                return this.translations.stopped;
            },

            clientText() {
                if (this.status.telemetry.client_address) {
                    return this.status.telemetry.client_address;
                }
                if (Array.isArray(this.status.telemetry.priority_clients)) {
                    const first = this.status.telemetry.priority_clients[0];
                    if (first) return first;
                }
                return this.translations.noClient;
            },

            uptimeText() {
                if (this.status.telemetry.uptime_seconds !== null) {
                    return formatDuration(this.status.telemetry.uptime_seconds);
                }
                if (this.status.updatedAt) {
                    const elapsed = Math.max(0, (Date.now() - this.status.updatedAt) / 1000);
                    return formatDuration(elapsed);
                }
                return this.translations.unknown;
            },

            lastUpdatedText() {
                return formatTimestamp(this.status.updatedAt);
            },

            sampleRateText() {
                const rate = this.status.telemetry.last_header?.sample_rate || 0;
                return rate > 0 ? `${formatNumber(rate)} Hz` : this.translations.unknown;
            },

            channelsText() {
                const channels = this.status.telemetry.last_header?.channels || 0;
                return channels > 0 ? formatNumber(channels) : this.translations.unknown;
            },

            bitDepthText() {
                const fmt = this.status.telemetry.last_header?.format;
                const depth = bitDepthFromFormat(fmt);
                if (depth) return `${depth}-bit`;
                return fmt ? fmt : this.translations.unknown;
            },

            latencyText() {
                const rate = this.status.telemetry.last_header?.sample_rate || 0;
                const frames = this.status.telemetry.buffered_frames || 0;
                if (rate > 0 && frames >= 0) {
                    const ms = (frames / rate) * 1000;
                    return `${ms.toFixed(2)} ms`;
                }
                return this.translations.unknown;
            },

            throughputText() {
                const rate = this.status.telemetry.last_header?.sample_rate || 0;
                const channels = this.status.telemetry.last_header?.channels || 0;
                const bps = bytesPerSampleFromFormat(
                    this.status.telemetry.last_header?.format
                );
                if (rate > 0 && channels > 0 && bps) {
                    const bytesPerSec = rate * channels * bps;
                    return `${formatBytes(bytesPerSec)}/s`;
                }
                return this.translations.unknown;
            },
        };
    }

    window.createTcpInputPage = createTcpInputPage;
    window.formatNumber = formatNumber;
    window.formatBytes = formatBytes;
    window.formatDuration = formatDuration;
    window.formatTimestamp = formatTimestamp;
})();
