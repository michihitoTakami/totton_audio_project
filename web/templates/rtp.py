"""HTML template for RTP session management UI."""


def get_rtp_sessions_html() -> str:
    """Return RTP session management HTML."""
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RTPセッション管理 | Magic Box</title>
    <style>
        :root {
            color-scheme: dark;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            padding: 24px;
            background: #111328;
            color: #f5f5f7;
        }
        a {
            color: #7dd3fc;
            text-decoration: none;
        }
        a:hover { text-decoration: underline; }
        h1 {
            font-size: 1.6rem;
            margin: 0 0 4px;
        }
        h2 {
            font-size: 1rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #94a3b8;
            margin: 0 0 12px;
        }
        p {
            margin: 0 0 12px;
            color: #cbd5f5;
        }
        .page-header {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 20px;
        }
        .page-header nav {
            display: flex;
            gap: 12px;
            font-size: 0.9rem;
        }
        .card {
            background: #181b36;
            border: 1px solid #1f2342;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(10, 14, 35, 0.35);
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .form-group label {
            font-size: 0.85rem;
            color: #94a3b8;
        }
        input[type="text"],
        input[type="number"],
        textarea,
        select {
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #242849;
            background: #10122a;
            color: #f8fafc;
            font-size: 0.95rem;
            outline: none;
            transition: border 0.2s;
        }
        textarea {
            min-height: 120px;
            resize: vertical;
        }
        input:focus,
        textarea:focus,
        select:focus {
            border-color: #38bdf8;
        }
        .field-error {
            display: none;
            font-size: 0.75rem;
            color: #f87171;
        }
        .form-group.has-error input,
        .form-group.has-error textarea {
            border-color: #f97316;
        }
        .sync-presets {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px;
        }
        .preset-card {
            border-radius: 12px;
            border: 1px solid #242849;
            background: #11142c;
            padding: 12px;
            cursor: pointer;
            transition: border 0.2s, background 0.2s;
        }
        .preset-card input {
            display: none;
        }
        .preset-card .preset-title {
            font-weight: 600;
            color: #e2e8f0;
        }
        .preset-card .preset-desc {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 4px;
        }
        .preset-card.active {
            border-color: #38bdf8;
            background: #0b1223;
        }
        .btn-row {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 16px;
        }
        button {
            border: none;
            border-radius: 10px;
            padding: 12px 18px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s, transform 0.2s;
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .btn-primary {
            background: linear-gradient(90deg, #38bdf8, #6366f1);
            color: #0b1223;
        }
        .btn-secondary {
            background: transparent;
            border: 1px solid #2f3354;
            color: #e2e8f0;
        }
        .btn-danger {
            background: #ef4444;
            color: #fff;
        }
        .section-desc {
            font-size: 0.85rem;
            color: #94a3b8;
            margin-bottom: 18px;
        }
        .session-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-top: 12px;
        }
        .session-card {
            border: 1px solid #242849;
            border-radius: 12px;
            padding: 16px;
            background: #10122a;
        }
        .session-card-header {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 12px;
        }
        .session-id {
            font-size: 1.05rem;
            font-weight: 600;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
        }
        .status-ok { background: rgba(34,197,94,0.15); color: #4ade80; }
        .status-warn { background: rgba(249,115,22,0.15); color: #fb923c; }
        .status-error { background: rgba(248,113,113,0.15); color: #f87171; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
        }
        .metric {
            padding: 10px 12px;
            border-radius: 10px;
            background: #0c0f24;
            border: 1px solid #1e213f;
        }
        .metric-label {
            font-size: 0.75rem;
            color: #94a3b8;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1rem;
            font-weight: 600;
            color: #e2e8f0;
        }
        .list-header {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }
        .list-meta {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.8rem;
            color: #94a3b8;
        }
        .badge {
            padding: 2px 8px;
            border-radius: 999px;
            background: rgba(99,102,241,0.2);
            color: #c7d2fe;
            font-size: 0.75rem;
        }
        .empty-state {
            padding: 32px;
            text-align: center;
            color: #94a3b8;
            border: 1px dashed #2a2f52;
            border-radius: 12px;
            margin-top: 12px;
        }
        .toast-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 999;
        }
        .toast {
            min-width: 240px;
            padding: 12px 16px;
            border-radius: 10px;
            color: #0f172a;
            font-weight: 600;
            animation: fade-in 0.3s ease;
        }
        .toast-success { background: #34d399; }
        .toast-error { background: #f87171; }
        .toast-info { background: #38bdf8; color: #082f49; }
        @keyframes fade-in {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .linear-progress {
            position: relative;
            width: 100%;
            height: 3px;
            overflow: hidden;
            border-radius: 999px;
            background: rgba(59,130,246,0.2);
            margin-top: 12px;
            display: none;
        }
        .linear-progress.visible { display: block; }
        .linear-progress::after {
            content: "";
            position: absolute;
            left: -40%;
            width: 40%;
            height: 100%;
            background: linear-gradient(90deg, transparent, #38bdf8, transparent);
            animation: progress-slide 1s infinite;
        }
        @keyframes progress-slide {
            0% { left: -40%; }
            100% { left: 100%; }
        }
        @media (max-width: 640px) {
            body { padding: 16px; }
            .form-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <header class="page-header">
        <div>
            <h1>RTPセッション管理</h1>
            <p>Control Plane APIをGUIで操作し、RTPエンドポイントの切り替え／監視を行います。</p>
        </div>
        <nav>
            <a href="/">ユーザーページ</a>
            <a href="/admin">管理者ページ</a>
            <a href="https://github.com/michihitoTakami/michy_os/tree/main/docs/architecture/rtp_session_manager.md" target="_blank">ドキュメント</a>
        </nav>
    </header>

    <main>
        <section class="card">
            <h2>New Session</h2>
            <p class="section-desc">SDPの貼り付け、IP/ポート、同期モード、SRTPキーの入力を行い <strong>POST /api/rtp/sessions</strong> へ送信します。</p>
            <form id="sessionForm" autocomplete="off">
                <div class="form-grid">
                    <div class="form-group" data-field="sessionId">
                        <label for="sessionId">セッションID</label>
                        <input id="sessionId" name="sessionId" type="text" placeholder="aes67-main" maxlength="64" required>
                        <div class="field-error" id="sessionIdError"></div>
                    </div>
                    <div class="form-group" data-field="bindAddress">
                        <label for="bindAddress">バインドIP</label>
                        <input id="bindAddress" name="bindAddress" type="text" placeholder="0.0.0.0" required>
                        <div class="field-error" id="bindAddressError"></div>
                    </div>
                    <div class="form-group" data-field="port">
                        <label for="port">ポート</label>
                        <input id="port" name="port" type="number" min="1" max="65535" placeholder="6000" required>
                        <div class="field-error" id="portError"></div>
                    </div>
                    <div class="form-group" data-field="sourceHost">
                        <label for="sourceHost">ソースIPフィルタ (任意)</label>
                        <input id="sourceHost" name="sourceHost" type="text" placeholder="送信元IPを固定する場合">
                        <div class="field-error" id="sourceHostError"></div>
                    </div>
                </div>

                <div style="margin:20px 0 12px;">
                    <label style="display:block; font-size:0.85rem; color:#94a3b8; margin-bottom:8px;">同期モード</label>
                    <div class="sync-presets" id="syncPresetGroup">
                        <label class="preset-card active" data-preset="low">
                            <input type="radio" name="syncMode" value="low" checked>
                            <div class="preset-title">低遅延 (5ms)</div>
                            <div class="preset-desc">リスニング用途。PTPなし、watchdog 0.5s。</div>
                        </label>
                        <label class="preset-card" data-preset="stable">
                            <input type="radio" name="syncMode" value="stable">
                            <div class="preset-title">安定優先 (20ms)</div>
                            <div class="preset-desc">ライブ配信受信等で使うゆとりバッファ。</div>
                        </label>
                        <label class="preset-card" data-preset="ptp">
                            <input type="radio" name="syncMode" value="ptp">
                            <div class="preset-title">PTP同期</div>
                            <div class="preset-desc">PTPロック必須。10msバッファ + PTP IF。</div>
                        </label>
                    </div>
                </div>

                <div class="form-grid" id="ptpFields" style="display:none;">
                    <div class="form-group" data-field="ptpInterface">
                        <label for="ptpInterface">PTPインターフェイス</label>
                        <input id="ptpInterface" name="ptpInterface" type="text" placeholder="例: eth0">
                        <div class="field-error" id="ptpInterfaceError"></div>
                    </div>
                </div>

                    <div class="form-group" style="margin-top:16px;" data-field="sdpBody">
                        <label for="sdpBody">SDP (任意)</label>
                        <textarea id="sdpBody" name="sdpBody" placeholder="ここに手動SDPを貼り付けると自動生成が上書きされます。"></textarea>
                        <div class="hint">`a=rtpmap` に <code>L16/48000/2</code> などを含めると、サンプルレート/チャネル/ビット深度/ペイロードタイプを自動で反映します。</div>
                        <div class="field-error" id="sdpBodyError"></div>
                    </div>

                <div style="margin-top:16px;">
                    <label style="display:flex; gap:8px; align-items:center; font-size:0.9rem;">
                        <input type="checkbox" id="srtpToggle">
                        <span>SRTPを使用する</span>
                    </label>
                    <div class="form-grid" id="srtpFields" style="display:none; margin-top:12px;">
                        <div class="form-group">
                            <label for="cryptoSuite">Crypto Suite</label>
                            <select id="cryptoSuite">
                                <option value="AES_CM_128_HMAC_SHA1_80">AES_CM_128_HMAC_SHA1_80</option>
                                <option value="AES_CM_128_HMAC_SHA1_32">AES_CM_128_HMAC_SHA1_32</option>
                            </select>
                        </div>
                        <div class="form-group" data-field="srtpKey">
                            <label for="srtpKey">Base64キー (40文字以上)</label>
                            <input id="srtpKey" name="srtpKey" type="text" placeholder="ZHVtbXlLZXlTdHJpbmdUZWFzZXI=">
                            <div class="field-error" id="srtpKeyError"></div>
                        </div>
                    </div>
                </div>

                <div class="btn-row">
                    <button type="submit" class="btn-primary" id="startSessionBtn">セッション開始</button>
                    <button type="button" class="btn-secondary" id="resetFormBtn">入力をクリア</button>
                </div>
                <div class="linear-progress" id="formProgress"></div>
            </form>
        </section>

        <section class="card">
            <div class="list-header">
                <div>
                    <h2>Active Sessions</h2>
                    <p class="section-desc">バックグラウンドポーラがキャッシュしたRTCPメトリクスを表示します。</p>
                </div>
                <div class="list-meta">
                    <span id="lastUpdated">最終更新: --</span>
                    <span class="badge" id="sessionCountBadge">0 sessions</span>
                    <button type="button" class="btn-secondary" id="refreshSessions">再取得</button>
                </div>
            </div>
            <div class="session-list" id="sessionList">
                <div class="empty-state" id="sessionListEmpty">セッションがまだありません。上のフォームから作成してください。</div>
            </div>
        </section>
    </main>

    <div class="toast-container" id="toastContainer"></div>

    <script>
        (function () {
            const API_BASE = '/api/rtp';
            const REFRESH_INTERVAL_MS = 5000;
            const SYNC_PRESETS = {
                low: { targetLatency: 5, watchdog: 500, telemetry: 1000, enablePtp: false },
                stable: { targetLatency: 20, watchdog: 1500, telemetry: 1500, enablePtp: false },
                ptp: { targetLatency: 10, watchdog: 700, telemetry: 1000, enablePtp: true, ptpDomain: 0 },
            };

            const elements = {
                form: document.getElementById('sessionForm'),
                sessionId: document.getElementById('sessionId'),
                bindAddress: document.getElementById('bindAddress'),
                port: document.getElementById('port'),
                sourceHost: document.getElementById('sourceHost'),
                ptpInterface: document.getElementById('ptpInterface'),
                sdpBody: document.getElementById('sdpBody'),
                srtpToggle: document.getElementById('srtpToggle'),
                srtpFields: document.getElementById('srtpFields'),
                srtpKey: document.getElementById('srtpKey'),
                cryptoSuite: document.getElementById('cryptoSuite'),
                presetCards: document.querySelectorAll('.preset-card'),
                ptpFields: document.getElementById('ptpFields'),
                startBtn: document.getElementById('startSessionBtn'),
                resetBtn: document.getElementById('resetFormBtn'),
                refreshBtn: document.getElementById('refreshSessions'),
                sessionList: document.getElementById('sessionList'),
                sessionEmpty: document.getElementById('sessionListEmpty'),
                toastContainer: document.getElementById('toastContainer'),
                lastUpdated: document.getElementById('lastUpdated'),
                sessionCount: document.getElementById('sessionCountBadge'),
                formProgress: document.getElementById('formProgress'),
            };

            const state = {
                sessions: [],
                pollTimer: null,
                isSubmitting: false,
            };

            const fieldValidators = {
                sessionId: (value) => {
                    if (!value) return 'セッションIDは必須です';
                    return /^[A-Za-z0-9._-]{1,64}$/.test(value) ? '' : '英数字・._-のみ、最大64文字です';
                },
                bindAddress: (value) => (isValidIPv4(value) ? '' : 'IPv4形式で入力してください'),
                port: (value) => {
                    const port = Number(value);
                    if (!Number.isInteger(port)) return '整数を入力してください';
                    return port >= 1 && port <= 65535 ? '' : '1〜65535の範囲で指定してください';
                },
                sourceHost: (value) => (value ? (isValidIPv4(value) ? '' : 'IPv4形式のみ') : ''),
                ptpInterface: (value) => (isPtpEnabled() && !value ? 'PTPモードでは必須です' : ''),
                srtpKey: (value) => {
                    if (!isSrtpEnabled()) return '';
                    if (!value) return 'SRTPキーを入力してください';
                    if (!/^[A-Za-z0-9+/=]+$/.test(value)) return 'Base64形式のみ利用できます';
                    return value.length >= 40 ? '' : '40文字以上のBase64文字列が必要です';
                },
            };

            const rtpApi = {
                async listSessions() {
                    return request(`${API_BASE}/sessions`);
                },
                async createSession(payload) {
                    return request(`${API_BASE}/sessions`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                    });
                },
                async deleteSession(sessionId) {
                    return request(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}`, {
                        method: 'DELETE',
                    });
                },
            };

            async function request(url, options = {}) {
                const response = await fetch(url, options);
                if (!response.ok) {
                    let detail = `HTTP ${response.status}`;
                    try {
                        const data = await response.json();
                        detail = data.detail || data.message || detail;
                    } catch (_) {
                        // best effort
                    }
                    throw new Error(detail);
                }
                if (response.status === 204) {
                    return {};
                }
                return response.json();
            }

            function isValidIPv4(value) {
                if (!value) return false;
                const parts = value.trim().split('.');
                if (parts.length !== 4) return false;
                return parts.every((part) => {
                    if (!/^[0-9]{1,3}$/.test(part)) return false;
                    const num = Number(part);
                    return num >= 0 && num <= 255;
                });
            }

            function escapeHtml(value) {
                return value.replace(/[&<>"']/g, (char) => ({
                    '&': '&amp;',
                    '<': '&lt;',
                    '>': '&gt;',
                    '"': '&quot;',
                    "'": '&#039;',
                }[char]));
            }

            function isSrtpEnabled() {
                return elements.srtpToggle.checked;
            }

            function isPtpEnabled() {
                const active = document.querySelector('input[name="syncMode"]:checked');
                return active ? active.value === 'ptp' : false;
            }

            function validateField(name) {
                const element = elements[name];
                if (!element || !fieldValidators[name]) {
                    return true;
                }
                const value = element.value.trim();
                const error = fieldValidators[name](value);
                setFieldError(name, error);
                return !error;
            }

            function setFieldError(name, message) {
                const group = document.querySelector(`[data-field="${name}"]`);
                if (!group) return;
                const errorEl = group.querySelector('.field-error');
                if (!errorEl) return;
                if (message) {
                    group.classList.add('has-error');
                    errorEl.textContent = message;
                    errorEl.style.display = 'block';
                } else {
                    group.classList.remove('has-error');
                    errorEl.textContent = '';
                    errorEl.style.display = 'none';
                }
            }

            function validateForm() {
                return ['sessionId', 'bindAddress', 'port', 'sourceHost', 'ptpInterface', 'srtpKey'].every((field) => validateField(field));
            }

            function buildPayload() {
                const syncMode = document.querySelector('input[name="syncMode"]:checked')?.value || 'low';
                const preset = SYNC_PRESETS[syncMode] || SYNC_PRESETS.low;
                const payload = {
                    session_id: elements.sessionId.value.trim(),
                    endpoint: {
                        bind_address: elements.bindAddress.value.trim(),
                        port: Number(elements.port.value),
                    },
                    format: {
                        sample_rate: 48000,
                        channels: 2,
                        bits_per_sample: 24,
                        payload_type: 97,
                        big_endian: true,
                        signed_samples: true,
                    },
                    sync: {
                        target_latency_ms: preset.targetLatency,
                        watchdog_timeout_ms: preset.watchdog,
                        telemetry_interval_ms: preset.telemetry,
                        enable_ptp: preset.enablePtp,
                    },
                    rtcp: { enable: true },
                    advanced: {
                        socket_buffer_bytes: 1048576,
                        mtu_bytes: 1500,
                    },
                };

                const sourceHost = elements.sourceHost.value.trim();
                if (sourceHost) {
                    payload.endpoint.source_host = sourceHost;
                }

                if (preset.enablePtp) {
                    payload.sync.ptp_interface = elements.ptpInterface.value.trim();
                    payload.sync.ptp_domain = preset.ptpDomain ?? 0;
                }

                const sdpBody = elements.sdpBody.value.trim();
                if (sdpBody) {
                    payload.sdp = { body: sdpBody };
                }

                if (isSrtpEnabled()) {
                    payload.security = {
                        crypto_suite: elements.cryptoSuite.value,
                        key_base64: elements.srtpKey.value.trim(),
                    };
                }

                return payload;
            }

            function renderSessions() {
                if (!state.sessions.length) {
                    elements.sessionEmpty.style.display = 'block';
                    elements.sessionList.querySelectorAll('.session-card').forEach((card) => card.remove());
                    elements.sessionCount.textContent = '0 sessions';
                    return;
                }
                elements.sessionEmpty.style.display = 'none';
                elements.sessionList.innerHTML = state.sessions
                    .map((session) => renderSessionCard(session))
                    .join('');
                elements.sessionCount.textContent = `${state.sessions.length} session${state.sessions.length > 1 ? 's' : ''}`;

                elements.sessionList.querySelectorAll('[data-stop-session]').forEach((btn) => {
                    btn.addEventListener('click', () => handleStopSession(btn.dataset.stopSession));
                });
            }

            function renderSessionCard(session) {
                const statusMeta = getSessionStatus(session);
                return `
                    <div class="session-card">
                        <div class="session-card-header">
                            <div class="session-id">${escapeHtml(session.session_id || 'unknown')}</div>
                            <span class="status-pill ${statusMeta.className}">
                                <span class="dot"></span>${statusMeta.label}
                            </span>
                        </div>
                        <div class="metrics-grid">
                            ${renderMetric('Packets', formatNumber(session.packets_received) + ' pkts')}
                            ${renderMetric('Late Packets', formatNumber(session.late_packets || 0))}
                            ${renderMetric('RTCP Delay', formatLatency(session.avg_transit_usec))}
                            ${renderMetric('Jitter', formatLatency(session.network_jitter_usec))}
                            ${renderMetric('PTP', session.ptp_locked ? 'LOCKED' : 'Unlocked')}
                            ${renderMetric('Last Packet', formatRelativeTime(session.last_packet_unix_ms))}
                        </div>
                        <div class="btn-row" style="margin-top:16px;">
                            <button class="btn-secondary" type="button" data-stop-session="${escapeHtml(session.session_id || '')}">停止</button>
                        </div>
                    </div>
                `;
            }

            function renderMetric(label, value) {
                return `
                    <div class="metric">
                        <div class="metric-label">${label}</div>
                        <div class="metric-value">${value}</div>
                    </div>
                `;
            }

            function getSessionStatus(session) {
                const now = Date.now();
                const lastPacket = session.last_packet_unix_ms || session.updated_at_unix_ms;
                if (!session.packets_received) {
                    return { label: '待機中', className: 'status-warn' };
                }
                if (lastPacket && now - lastPacket > 8000) {
                    return { label: '更新停止', className: 'status-error' };
                }
                if (session.late_packets && session.late_packets > 0) {
                    return { label: '遅延あり', className: 'status-warn' };
                }
                return { label: '接続中', className: 'status-ok' };
            }

            function formatNumber(value) {
                return new Intl.NumberFormat('ja-JP').format(value ?? 0);
            }

            function formatLatency(usec) {
                if (typeof usec !== 'number' || Number.isNaN(usec)) return '--';
                const ms = usec / 1000;
                return ms.toFixed(ms >= 10 ? 1 : 2) + ' ms';
            }

            function formatRelativeTime(timestamp) {
                if (!timestamp) return '--';
                const diff = Date.now() - timestamp;
                if (diff < 0) return '0 ms';
                if (diff < 1000) return 'live';
                if (diff < 60000) return Math.round(diff / 1000) + ' s';
                return Math.round(diff / 60000) + ' m';
            }

            function showToast(message, type = 'info') {
                const toast = document.createElement('div');
                toast.className = `toast toast-${type}`;
                toast.textContent = message;
                elements.toastContainer.appendChild(toast);
                setTimeout(() => toast.remove(), 3500);
            }

            async function fetchSessions() {
                toggleRefreshLoading(true);
                try {
                    const data = await rtpApi.listSessions();
                    state.sessions = data.sessions || [];
                    if (data.polled_at_unix_ms) {
                        const date = new Date(data.polled_at_unix_ms);
                        elements.lastUpdated.textContent = '最終更新: ' + date.toLocaleTimeString('ja-JP');
                    } else {
                        elements.lastUpdated.textContent = '最終更新: now';
                    }
                    renderSessions();
                } catch (error) {
                    showToast('セッション一覧の取得に失敗しました: ' + error.message, 'error');
                } finally {
                    toggleRefreshLoading(false);
                }
            }

            function toggleRefreshLoading(loading) {
                if (!elements.refreshBtn) return;
                elements.refreshBtn.disabled = loading;
                elements.refreshBtn.textContent = loading ? '更新中...' : '再取得';
            }

            function toggleFormLoading(loading) {
                state.isSubmitting = loading;
                elements.startBtn.disabled = loading;
                elements.formProgress.classList.toggle('visible', loading);
                elements.startBtn.textContent = loading ? '送信中...' : 'セッション開始';
            }

            async function handleStopSession(sessionId) {
                if (!sessionId) return;
                try {
                    const confirmed = window.confirm(`${sessionId} を停止しますか？`);
                    if (!confirmed) return;
                    showToast(`${sessionId} の停止を送信中...`, 'info');
                    await rtpApi.deleteSession(sessionId);
                    showToast(`${sessionId} を停止しました`, 'success');
                    fetchSessions();
                } catch (error) {
                    showToast('停止に失敗しました: ' + error.message, 'error');
                }
            }

            function resetForm() {
                elements.form.reset();
                elements.sessionId.value = 'aes67-main';
                elements.bindAddress.value = '0.0.0.0';
                elements.port.value = 6000;
                elements.srtpFields.style.display = 'none';
                elements.presetCards.forEach((card) => {
                    const isLow = card.dataset.preset === 'low';
                    card.classList.toggle('active', isLow);
                    card.querySelector('input').checked = isLow;
                });
                elements.ptpFields.style.display = 'none';
                ['sessionId', 'bindAddress', 'port', 'sourceHost', 'ptpInterface', 'srtpKey'].forEach((field) => setFieldError(field, ''));
            }

            function attachEventListeners() {
                elements.form.addEventListener('submit', async (event) => {
                    event.preventDefault();
                    if (!validateForm()) {
                        showToast('入力内容を確認してください', 'error');
                        return;
                    }
                    toggleFormLoading(true);
                    try {
                        const payload = buildPayload();
                        const response = await rtpApi.createSession(payload);
                        showToast(response.message || 'セッションを開始しました', 'success');
                        fetchSessions();
                    } catch (error) {
                        showToast('セッション開始に失敗しました: ' + error.message, 'error');
                    } finally {
                        toggleFormLoading(false);
                    }
                });

                elements.resetBtn.addEventListener('click', resetForm);
                elements.refreshBtn.addEventListener('click', fetchSessions);

                ['sessionId', 'bindAddress', 'port', 'sourceHost', 'ptpInterface', 'srtpKey'].forEach((field) => {
                    const element = elements[field];
                    if (!element) return;
                    element.addEventListener('input', () => validateField(field));
                });

                elements.srtpToggle.addEventListener('change', () => {
                    const enabled = isSrtpEnabled();
                    elements.srtpFields.style.display = enabled ? 'grid' : 'none';
                    if (!enabled) {
                        setFieldError('srtpKey', '');
                    }
                });

                elements.presetCards.forEach((card) => {
                    card.addEventListener('click', () => {
                        elements.presetCards.forEach((c) => c.classList.remove('active'));
                        card.classList.add('active');
                        const radio = card.querySelector('input');
                        radio.checked = true;
                        elements.ptpFields.style.display = radio.value === 'ptp' ? 'grid' : 'none';
                    });
                });
            }

            function init() {
                resetForm();
                attachEventListeners();
                fetchSessions();
                state.pollTimer = setInterval(fetchSessions, REFRESH_INTERVAL_MS);
            }

            init();
        })();
    </script>
</body>
</html>
"""


"""HTML template for RTP session management UI."""
