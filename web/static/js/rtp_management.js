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

        // RTP Configuration (Latency only)
        rtpConfig: {
            latency_preset: '50',  // Default: Normal (50ms)
            target_latency_ms: 50
        },
        applyingConfig: false,

        // Polling interval
        pollingInterval: null,

        /**
         * Initialize the page
         */
        async init() {
            await this.loadActiveSessions();
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
                    showToast(t('rtp.scan.success'), 'success');
                } else {
                    const error = await response.json();
                    showToast(error.detail || t('rtp.scan.error'), 'error');
                }
            } catch (error) {
                console.error('Scan error:', error);
                showToast(t('rtp.scan.error'), 'error');
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
                    showToast(t('rtp.connect.success'), 'success');
                    await this.loadActiveSessions();
                    // Mark as existing session in scan results
                    stream.existing_session = true;
                } else {
                    const error = await response.json();
                    showToast(error.detail || t('rtp.connect.error'), 'error');
                }
            } catch (error) {
                console.error('Connect error:', error);
                showToast(t('rtp.connect.error'), 'error');
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
            if (!confirm(t('rtp.sessions.confirm_stop'))) {
                return;
            }

            try {
                const response = await fetch(`/api/rtp/sessions/${sessionId}`, {
                    method: 'DELETE'
                });

                if (response.ok) {
                    showToast(t('rtp.sessions.stopped'), 'success');
                    if (this.selectedSessionId === sessionId) {
                        this.selectedSessionId = null;
                        this.selectedSession = null;
                    }
                    await this.loadActiveSessions();
                } else {
                    const error = await response.json();
                    showToast(error.detail || t('rtp.sessions.stop_error'), 'error');
                }
            } catch (error) {
                console.error('Stop session error:', error);
                showToast(t('rtp.sessions.stop_error'), 'error');
            }
        },

        /**
         * Apply latency configuration to all active sessions
         */
        async applyLatencyConfig() {
            this.applyingConfig = true;
            try {
                // 1. Get current active sessions
                const currentSessions = [...this.sessions];
                if (currentSessions.length === 0) {
                    showToast(t('rtp.config.saved_no_restart'), 'success');
                    return;
                }

                // 2. Stop all sessions
                for (const session of currentSessions) {
                    try {
                        await fetch(`/api/rtp/sessions/${session.session_id}`, {
                            method: 'DELETE'
                        });
                    } catch (error) {
                        console.error(`Failed to stop session ${session.session_id}:`, error);
                    }
                }

                // 3. Wait a bit for sessions to stop
                await new Promise(resolve => setTimeout(resolve, 500));

                // 4. Restart sessions with new latency
                let restartedCount = 0;
                for (const session of currentSessions) {
                    try {
                        const restartPayload = {
                            session_id: session.session_id,
                            port: session.port,
                            bind_address: session.bind_address || '0.0.0.0',
                            target_latency_ms: parseInt(this.rtpConfig.target_latency_ms)
                        };

                        if (session.sample_rate) restartPayload.sample_rate = session.sample_rate;
                        if (session.channels) restartPayload.channels = session.channels;
                        if (session.payload_type) restartPayload.payload_type = session.payload_type;

                        const restartResponse = await fetch('/api/rtp/sessions', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(restartPayload)
                        });

                        if (restartResponse.ok) {
                            restartedCount++;
                        }
                    } catch (error) {
                        console.error(`Failed to restart session ${session.session_id}:`, error);
                    }
                }

                // 5. Reload sessions
                await this.loadActiveSessions();

                showToast(t('rtp.config.applied', {count: restartedCount, total: currentSessions.length}), 'success');
            } catch (error) {
                console.error('Apply config error:', error);
                showToast(t('rtp.config.apply_error'), 'error');
            } finally {
                this.applyingConfig = false;
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
