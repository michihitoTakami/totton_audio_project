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

        buildSessionCreateRequest(base, overrideLatencyMs = null) {
            const toInt = (value, fallback) => {
                const parsed = parseInt(value, 10);
                return Number.isFinite(parsed) ? parsed : fallback;
            };

            const payload = {
                session_id: base.session_id,
                endpoint: {
                    bind_address: base.bind_address || '0.0.0.0',
                    port: toInt(base.port, 6000),
                },
                format: {
                    sample_rate: toInt(base.sample_rate, 48000),
                    channels: toInt(base.channels, 2),
                    bits_per_sample: toInt(base.bits_per_sample, 24),
                    payload_type: toInt(base.payload_type, 97),
                    big_endian: base.big_endian ?? true,
                    signed_samples: base.signed ?? true,
                },
                sync: {
                    target_latency_ms: toInt(
                        overrideLatencyMs ?? base.target_latency_ms,
                        toInt(this.rtpConfig.target_latency_ms, 50)
                    ),
                    watchdog_timeout_ms: toInt(base.watchdog_timeout_ms, 500),
                    telemetry_interval_ms: toInt(base.telemetry_interval_ms, 1000),
                    enable_ptp: base.enable_ptp ?? false,
                },
                rtcp: {
                    enable: base.enable_rtcp ?? true,
                },
            };

            if (base.source_host) {
                payload.endpoint.source_host = base.source_host;
            }
            if (base.multicast) {
                payload.endpoint.multicast = true;
                if (base.multicast_group) {
                    payload.endpoint.multicast_group = base.multicast_group;
                }
            }
            if (base.interface) {
                payload.endpoint.interface = base.interface;
            }
            const ttl = toInt(base.ttl, null);
            if (ttl !== null) {
                payload.endpoint.ttl = ttl;
            }
            const dscp = toInt(base.dscp, null);
            if (dscp !== null) {
                payload.endpoint.dscp = dscp;
            }
            const rtcpPort = toInt(base.rtcp_port, null);
            if (rtcpPort !== null) {
                payload.rtcp.port = rtcpPort;
            }
            if (payload.sync.enable_ptp && base.ptp_interface) {
                payload.sync.ptp_interface = base.ptp_interface;
            }
            const ptpDomain = toInt(base.ptp_domain, null);
            if (payload.sync.enable_ptp && ptpDomain !== null) {
                payload.sync.ptp_domain = ptpDomain;
            }

            return payload;
        },

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
                const payload = this.buildSessionCreateRequest(
                    {
                        session_id: stream.session_id,
                        bind_address: stream.bind_address || '0.0.0.0',
                        port: stream.port,
                        source_host: stream.source_host,
                        multicast: stream.multicast,
                        multicast_group: stream.multicast_group,
                        interface: stream.interface,
                        ttl: stream.ttl,
                        dscp: stream.dscp,
                        sample_rate: stream.sample_rate,
                        channels: stream.channels,
                        bits_per_sample: stream.bits_per_sample,
                        payload_type: stream.payload_type,
                        big_endian: stream.big_endian,
                        signed: stream.signed,
                        enable_rtcp: stream.enable_rtcp,
                        rtcp_port: stream.rtcp_port,
                        enable_ptp: stream.enable_ptp,
                        ptp_interface: stream.ptp_interface,
                        ptp_domain: stream.ptp_domain,
                    },
                    stream.target_latency_ms ?? this.rtpConfig.target_latency_ms
                );

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
                const failedSessions = [];
                for (const session of currentSessions) {
                    try {
                        const restartPayload = this.buildSessionCreateRequest(
                            session,
                            this.rtpConfig.target_latency_ms
                        );

                        const restartResponse = await fetch('/api/rtp/sessions', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(restartPayload)
                        });

                        if (restartResponse.ok) {
                            restartedCount++;
                        } else {
                            failedSessions.push(session.session_id);
                            const error = await restartResponse.json().catch(() => ({}));
                            console.error(`Failed to restart session ${session.session_id}:`, error);
                        }
                    } catch (error) {
                        console.error(`Failed to restart session ${session.session_id}:`, error);
                        failedSessions.push(session.session_id);
                    }
                }

                // 5. Reload sessions
                await this.loadActiveSessions();

                if (failedSessions.length === 0) {
                    showToast(t('rtp.config.applied', {count: restartedCount, total: currentSessions.length}), 'success');
                } else {
                    showToast(t('rtp.config.apply_error'), 'error');
                }
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
