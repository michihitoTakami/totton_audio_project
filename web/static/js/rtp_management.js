/**
 * RTP Management Page - Alpine.js Data Handler
 */

function rtpManagementData() {
    return {
        // RTP Stream Discovery
        scanning: false,
        scanResult: {
            streams: [],
            scanned_at_unix_ms: null,
            duration_ms: null
        },

        // Active RTP Sessions
        sessions: [],
        selectedSessionId: null,
        selectedSession: null,

        // RTP Configuration
        rtpConfig: {
            port: null,
            bind_address: null,
            payload_type: null
        },
        savingConfig: false,

        // Polling interval
        pollingInterval: null,

        /**
         * Initialize the page
         */
        async init() {
            await this.loadActiveSessions();
            await this.loadRtpConfig();
            this.startPolling();
        },

        /**
         * Scan for available RTP streams
         */
        async scanStreams() {
            this.scanning = true;
            try {
                const response = await fetch('/api/rtp/discover');
                if (response.ok) {
                    const data = await response.json();
                    this.scanResult = data;
                    showToast('Scan completed', 'success');
                } else {
                    const error = await response.json();
                    showToast(error.detail || 'Failed to scan RTP streams', 'error');
                }
            } catch (error) {
                console.error('Scan error:', error);
                showToast('Failed to scan RTP streams', 'error');
            } finally {
                this.scanning = false;
            }
        },

        /**
         * Connect to a discovered RTP stream
         */
        async connectStream(stream) {
            try {
                const payload = {
                    session_id: stream.session_id,
                    port: stream.port,
                    bind_address: stream.bind_address || '0.0.0.0'
                };

                if (stream.sample_rate) {
                    payload.sample_rate = stream.sample_rate;
                }
                if (stream.channels) {
                    payload.channels = stream.channels;
                }
                if (stream.payload_type) {
                    payload.payload_type = stream.payload_type;
                }
                if (stream.multicast) {
                    payload.multicast = true;
                    payload.multicast_group = stream.multicast_group;
                }

                const response = await fetch('/api/rtp/sessions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (response.ok) {
                    showToast('Connected to stream', 'success');
                    await this.loadActiveSessions();
                    // Mark as existing session in scan results
                    stream.existing_session = true;
                } else {
                    const error = await response.json();
                    showToast(error.detail || 'Failed to connect to stream', 'error');
                }
            } catch (error) {
                console.error('Connect error:', error);
                showToast('Failed to connect to stream', 'error');
            }
        },

        /**
         * Load active RTP sessions
         */
        async loadActiveSessions() {
            try {
                const response = await fetch('/api/rtp/sessions');
                if (response.ok) {
                    const data = await response.json();
                    this.sessions = data.sessions || [];

                    // Update selected session if it still exists
                    if (this.selectedSessionId) {
                        this.selectedSession = this.sessions.find(s => s.session_id === this.selectedSessionId);
                    }
                }
            } catch (error) {
                console.error('Load sessions error:', error);
            }
        },

        /**
         * Select a session for telemetry viewing
         */
        selectSession(sessionId) {
            this.selectedSessionId = sessionId;
            this.selectedSession = this.sessions.find(s => s.session_id === sessionId);
        },

        /**
         * Stop an RTP session
         */
        async stopSession(sessionId) {
            if (!confirm('Stop this RTP session?')) {
                return;
            }

            try {
                const response = await fetch(`/api/rtp/sessions/${sessionId}`, {
                    method: 'DELETE'
                });

                if (response.ok) {
                    showToast('Session stopped', 'success');
                    if (this.selectedSessionId === sessionId) {
                        this.selectedSessionId = null;
                        this.selectedSession = null;
                    }
                    await this.loadActiveSessions();
                } else {
                    const error = await response.json();
                    showToast(error.detail || 'Failed to stop session', 'error');
                }
            } catch (error) {
                console.error('Stop session error:', error);
                showToast('Failed to stop session', 'error');
            }
        },

        /**
         * Load RTP configuration from config.json
         */
        async loadRtpConfig() {
            // Load current config from the API (if available)
            // For now, initialize with default values
            this.rtpConfig = {
                port: 46000,
                bind_address: '0.0.0.0',
                payload_type: 96
            };
        },

        /**
         * Save RTP configuration
         */
        async saveRtpConfig() {
            this.savingConfig = true;
            try {
                const payload = {
                    port: parseInt(this.rtpConfig.port),
                    bind_address: this.rtpConfig.bind_address,
                    payload_type: parseInt(this.rtpConfig.payload_type)
                };

                const response = await fetch('/api/rtp/config', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (response.ok) {
                    const data = await response.json();
                    showToast(data.message || 'Configuration saved', 'success');
                } else {
                    const error = await response.json();
                    showToast(error.detail || 'Failed to save configuration', 'error');
                }
            } catch (error) {
                console.error('Save config error:', error);
                showToast('Failed to save configuration', 'error');
            } finally {
                this.savingConfig = false;
            }
        },

        /**
         * Start polling for session updates
         */
        startPolling() {
            this.pollingInterval = setInterval(() => {
                this.loadActiveSessions();
            }, 2000); // Poll every 2 seconds
        },

        /**
         * Format timestamp (Unix ms) to human-readable string
         */
        formatTimestamp(unixMs) {
            if (!unixMs) return '-';
            const date = new Date(unixMs);
            return date.toLocaleString();
        },

        /**
         * Format bytes to human-readable string
         */
        formatBytes(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return (bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i];
        }
    };
}
