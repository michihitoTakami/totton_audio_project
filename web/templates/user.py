"""User-facing HTML template."""


def get_embedded_html() -> str:
    """Return embedded HTML UI"""
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Upsampler</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
            max-width: 500px;
            margin: 0 auto;
        }
        h1 { color: #00d4ff; margin-bottom: 20px; font-size: 1.4em; }
        h2 { color: #888; font-size: 11px; margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 1px; }
        .card {
            background: #16213e;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .status-item {
            text-align: center;
            flex: 1;
        }
        .status-item .label { font-size: 10px; color: #666; text-transform: uppercase; }
        .status-item .value { font-size: 14px; font-weight: 600; margin-top: 2px; }
        .status-item.ok .value { color: #00ff88; }
        .status-item.error .value { color: #ff4444; }
        .form-group { margin-bottom: 12px; }
        .form-group label {
            display: block;
            margin-bottom: 6px;
            color: #aaa;
            font-size: 13px;
        }
        .form-group select {
            width: 100%;
            padding: 12px;
            border: 1px solid #0f3460;
            border-radius: 6px;
            background: #0f3460;
            color: #eee;
            font-size: 14px;
            cursor: pointer;
        }
        .form-group select:focus {
            outline: none;
            border-color: #00d4ff;
        }
        .btn-row { display: flex; gap: 10px; margin-top: 16px; }
        button {
            flex: 1;
            padding: 12px 20px;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn-primary { background: #00d4ff; color: #000; }
        .btn-primary:hover { background: #00a8cc; }
        .btn-primary:disabled { background: #555; color: #888; cursor: not-allowed; }
        .btn-secondary { background: #0f3460; color: #eee; }
        .btn-secondary:hover { background: #1a4b7c; }
        .btn-secondary:disabled { background: #333; color: #666; cursor: not-allowed; }
        .message {
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            display: none;
            font-size: 13px;
            text-align: center;
        }
        .message.success { background: #00ff8840; display: block; }
        .message.error { background: #ff444440; display: block; }
        .rtp-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            color: #888;
            margin-bottom: 8px;
        }
        .rtp-details {
            margin-top: 12px;
            padding: 10px;
            background: #0f3460;
            border-radius: 6px;
            font-size: 12px;
            line-height: 1.5;
        }
        .rtp-details strong {
            color: #fff;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: #1f4068;
            color: #9ec9ff;
            margin-right: 6px;
        }
        .pill-active {
            background: #00ff8840;
            color: #00ff88;
        }
        .pill-auto {
            background: #ffd54f33;
            color: #ffd54f;
        }
        .rtp-active {
            margin-top: 10px;
            font-size: 12px;
            color: #9ec9ff;
        }
        .rtp-active span {
            font-weight: 600;
            color: #fff;
        }
        .rtp-scan-row {
            display: flex;
            gap: 8px;
            align-items: center;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }
        .rtp-scan-row button {
            flex: 0 0 auto;
        }
        .scan-status {
            font-size: 11px;
            color: #666;
        }
        .warning-banner {
            background: #ffaa0030;
            border: 1px solid #ffaa00;
            border-radius: 4px;
            padding: 8px 12px;
            margin-top: 8px;
            font-size: 12px;
            color: #ffaa00;
            display: none;
        }
        .warning-banner.visible { display: block; }
        .rtp-note {
            margin-top: 8px;
            font-size: 12px;
            color: #ffd54f;
        }
        /* Crossfeed Toggle Switch */
        .toggle-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 0;
        }
        .toggle-switch {
            position: relative;
            width: 50px;
            height: 26px;
            background: #0f3460;
            border-radius: 13px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .toggle-switch.active {
            background: #00d4ff;
        }
        .toggle-switch::after {
            content: '';
            position: absolute;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: #eee;
            top: 2px;
            left: 2px;
            transition: transform 0.3s;
        }
        .toggle-switch.active::after {
            transform: translateX(24px);
        }
        .toggle-label {
            font-size: 13px;
            color: #aaa;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        /* Head Size Buttons */
        .head-size-group {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }
        .head-size-btn {
            flex: 1;
            padding: 10px 8px;
            border: 1px solid #0f3460;
            border-radius: 6px;
            background: #0f3460;
            color: #aaa;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
        }
        .head-size-btn:hover {
            background: #1a4b7c;
            border-color: #00d4ff;
        }
        .head-size-btn.active {
            background: #00d4ff;
            color: #000;
            border-color: #00d4ff;
        }
        .head-size-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        /* Info Text */
        .info-text {
            font-size: 12px;
            color: #666;
            margin-top: 12px;
            line-height: 1.5;
        }
        .info-text .icon {
            color: #00d4ff;
            margin-right: 4px;
        }
        /* Status Display */
        .status-display {
            margin-top: 12px;
            padding: 10px;
            background: #0f3460;
            border-radius: 6px;
            font-size: 12px;
        }
        .status-display .label {
            color: #666;
            margin-bottom: 4px;
        }
        .status-display .value {
            color: #eee;
            font-weight: 500;
        }
        .status-display .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #666;
            margin-right: 6px;
        }
        .status-display .status-indicator.active {
            background: #00ff88;
        }
        /* Loading Indicator */
        .loading-indicator {
            display: none;
            text-align: center;
            padding: 8px;
            font-size: 12px;
            color: #00d4ff;
        }
        .loading-indicator.visible {
            display: block;
        }
        .loading-indicator::after {
            content: '...';
            animation: dots 1.5s steps(4, end) infinite;
        }
        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60%, 100% { content: '...'; }
        }
        /* Input Mode */
        .input-mode-options {
            display: flex;
            gap: 12px;
        }
        .input-mode-option {
            flex: 1;
            padding: 12px;
            border: 1px solid #0f3460;
            border-radius: 6px;
            background: #0f3460;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            gap: 4px;
            transition: border-color 0.2s, background 0.2s;
        }
        .input-mode-option input[type="radio"] {
            appearance: none;
            width: 0;
            height: 0;
            margin: 0;
        }
        .input-mode-option .option-title {
            font-size: 14px;
            font-weight: 600;
            color: #fff;
        }
        .input-mode-option .option-desc {
            font-size: 12px;
            color: #888;
        }
        .input-mode-option.active {
            border-color: #00d4ff;
            background: #10294a;
        }
        .input-mode-option.disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .input-mode-info {
            margin-top: 12px;
            font-size: 12px;
            color: #9ec9ff;
        }
        .input-mode-info .note {
            color: #666;
            margin-top: 4px;
        }
    </style>
</head>
<body>
    <h1>GPU Upsampler</h1>

    <h2>Status</h2>
    <div class="card">
        <div class="status-row">
            <div class="status-item" id="daemonStatus">
                <div class="label">Daemon</div>
                <div class="value">-</div>
            </div>
            <div class="status-item" id="pwStatus">
                <div class="label">PipeWire</div>
                <div class="value">-</div>
            </div>
            <div class="status-item" id="eqStatus">
                <div class="label">EQ</div>
                <div class="value">-</div>
            </div>
        </div>
    </div>

    <h2>Input Mode</h2>
    <div class="card">
        <div class="input-mode-options" id="inputModeOptions">
            <label class="input-mode-option" data-mode="pipewire">
                <input type="radio" name="inputMode" value="pipewire">
                <span class="option-title">PipeWire</span>
                <span class="option-desc">ローカル入力（最小遅延）</span>
            </label>
            <label class="input-mode-option" data-mode="rtp">
                <input type="radio" name="inputMode" value="rtp">
                <span class="option-title">RTP</span>
                <span class="option-desc">ネットワーク入力</span>
            </label>
        </div>
        <div class="input-mode-info">
            <div id="inputModeStatusText">現在: -</div>
            <div class="note" id="inputModeRestartHint">切り替え時にデーモンを自動再起動します</div>
        </div>
        <div id="inputModeMessage" class="message"></div>
    </div>

    <h2>RTP Input</h2>
    <div class="card">
        <div class="rtp-scan-row">
            <button type="button" class="btn-secondary" id="rtpScanBtn">RTP入力をスキャン</button>
            <div class="scan-status" id="rtpScanStatus">未スキャン</div>
        </div>
        <div class="form-group">
            <label>受信候補</label>
            <select id="rtpStreamSelect" disabled>
                <option value="">スキャンしてください</option>
            </select>
        </div>
        <div class="btn-row">
            <button type="button" class="btn-primary" id="rtpStartBtn" disabled>開始</button>
            <button type="button" class="btn-secondary" id="rtpStopBtn" disabled>停止</button>
        </div>
        <div class="rtp-details" id="rtpDetails">
            スキャンして候補を読み込みます。ソースIPとポートを表示します。
        </div>
        <div class="rtp-active" id="rtpActiveSession">稼働中のRTPセッション: なし</div>
        <div id="rtpMessage" class="message"></div>
    </div>

    <h2>Output Device</h2>
    <div class="card">
        <form id="settingsForm">
            <div class="form-group">
                <select id="alsaDevice">
                    <option value="">Loading...</option>
                </select>
            </div>
            <div class="btn-row">
                <button type="submit" class="btn-primary" id="saveBtn">Save & Restart</button>
            </div>
        </form>
        <div id="settingsMessage" class="message"></div>
    </div>

    <h2>Phase Type</h2>
    <div class="card">
        <div class="form-group">
            <label>Filter Phase</label>
            <select id="phaseType">
                <option value="minimum" title="全帯域を最小位相で処理（最小レイテンシ）">Minimum Phase (推奨)</option>
                <option value="hybrid" title="100Hz以下は最小位相 / 100Hz以上は線形位相（10ms整列）">Hybrid Phase (100Hz以下最小 / 100Hz以上線形)</option>
            </select>
            <div style="font-size:12px; color:#9fb3ff; margin-top:8px; line-height:1.4;">
                ハイブリッドは100Hz以下を最小位相、100Hz以上を線形位相（10ms整列）で処理し、定位を維持しながら低域のプレリンギングを抑えます。
            </div>
        </div>
        <div id="phaseWarning" class="warning-banner">
            ⚠️ ハイブリッドは100Hz以下を最小位相、100Hz以上を線形位相（10ms整列）で処理します。約10msの整列ディレイが発生し、低遅延モードとは併用できません。
        </div>
        <div id="phaseMessage" class="message"></div>
    </div>

    <h2>Headphone EQ (OPRA)</h2>
    <div class="card">
        <div class="form-group">
            <label>Search Headphones</label>
            <input type="text" id="opraSearch" placeholder="e.g. HD650, DT770, AirPods..."
                   style="width:100%; padding:12px; border:1px solid #0f3460; border-radius:6px; background:#0f3460; color:#eee; font-size:14px;">
        </div>
        <div id="opraResults" style="max-height:200px; overflow-y:auto; margin-top:8px;"></div>
        <div id="opraSelected" style="display:none; margin-top:12px; padding:12px; background:#0f3460; border-radius:6px;">
            <div style="font-weight:600;" id="selectedName">-</div>
            <div style="font-size:12px; color:#888;" id="selectedVendor">-</div>
            <select id="opraEqSelect" style="width:100%; padding:8px; margin-top:8px; border:1px solid #16213e; border-radius:4px; background:#16213e; color:#eee;">
            </select>
            <label style="display:flex; align-items:center; gap:8px; margin-top:12px; cursor:pointer;">
                <input type="checkbox" id="modernTargetCheckbox" checked style="width:16px; height:16px; accent-color:#00d4ff;">
                <span style="font-size:13px;">Modern Target (KB5000_7)</span>
            </label>
            <div style="font-size:10px; color:#666; margin-left:24px;">最新のターゲットカーブに補正</div>
        </div>
        <div class="btn-row">
            <button type="button" class="btn-primary" id="applyOpraBtn" disabled>Apply EQ</button>
            <button type="button" class="btn-secondary" id="deactivateEqBtn">EQ Off</button>
        </div>
        <div id="opraMessage" class="message"></div>
        <div style="font-size:10px; color:#555; margin-top:12px; text-align:center;">
            EQ data: <a href="https://github.com/opra-project/OPRA" target="_blank" style="color:#00d4ff;">OPRA Project</a> (CC BY-SA 4.0)
        </div>
    </div>

    <h2>🎧 クロスフィード</h2>
    <div class="card">
        <div class="toggle-container">
            <div class="toggle-label">
                <span>クロスフィード</span>
            </div>
            <div class="toggle-switch" id="crossfeedToggle"></div>
        </div>
        <div class="form-group" style="margin-top: 16px;">
            <label>頭のサイズ:</label>
            <div class="head-size-group">
                <button type="button" class="head-size-btn" data-size="xs">XS</button>
                <button type="button" class="head-size-btn" data-size="s">S</button>
                <button type="button" class="head-size-btn" data-size="m" id="headSizeM">M</button>
                <button type="button" class="head-size-btn" data-size="l">L</button>
                <button type="button" class="head-size-btn" data-size="xl">XL</button>
            </div>
        </div>
        <div class="info-text">
            <span class="icon">ℹ️</span>正三角形配置（±30°）でスピーカーリスニングを再現
        </div>
        <div class="loading-indicator" id="crossfeedLoading">適用中</div>
        <div class="status-display" id="crossfeedStatus" style="display:none;">
            <div class="label">ステータス:</div>
            <div class="value">
                <span class="status-indicator" id="crossfeedStatusIndicator"></span>
                <span id="crossfeedStatusText">無効</span>
            </div>
        </div>
        <div id="crossfeedMessage" class="message"></div>
        <div style="font-size:10px; color:#555; margin-top:12px; text-align:center;">
            HRTF data: <a href="https://depositonce.tu-berlin.de/items/dc2a3076-a291-417e-97f0-7697e332c960" target="_blank" style="color:#00d4ff;">HUTUBS, TU Berlin</a> (CC BY 4.0)
        </div>
    </div>

    <script>
        const API = '';
        const RTP_API = '/api/rtp';
        const INPUT_MODE_API = '/api/input-mode';
        let currentAlsaDevice = '';
        let deviceList = [];
        let lastPipewireConnected = false;
        const inputModeState = {
            current: 'pipewire',
            switching: false,
        };
        const rtpState = {
            streams: [],
            selectedId: '',
            scanning: false,
            lastScanAt: null,
            activeSessions: [],
        };
        const inputModeOptions = document.querySelectorAll('.input-mode-option');
        const inputModeRadios = document.querySelectorAll('input[name="inputMode"]');
        const inputModeMessage = document.getElementById('inputModeMessage');
        const inputModeStatusText = document.getElementById('inputModeStatusText');
        const inputModeRestartHint = document.getElementById('inputModeRestartHint');
        const pwStatusLabel = document.querySelector('#pwStatus .label');
        const rtpSelect = document.getElementById('rtpStreamSelect');
        const rtpScanBtn = document.getElementById('rtpScanBtn');
        const rtpStartBtn = document.getElementById('rtpStartBtn');
        const rtpStopBtn = document.getElementById('rtpStopBtn');
        const rtpMessage = document.getElementById('rtpMessage');
        const rtpDetails = document.getElementById('rtpDetails');
        const rtpActiveSession = document.getElementById('rtpActiveSession');
        const rtpScanStatus = document.getElementById('rtpScanStatus');

        /**
         * Normalize port number to valid integer or null.
         * @param {any} value - Port number candidate
         * @returns {number|null} Normalized port or null if invalid
         */
        function normalizePort(value) {
            const parsed = Number(value);
            return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
        }

        /**
         * Convert RTP session metrics (from daemon telemetry) to discovery stream format.
         * This enables auto-started sessions to appear in the UI alongside scanned streams.
         * @param {Object} session - Session metrics from daemon
         * @returns {Object|null} Discovery stream object or null if invalid
         */
        function sessionMetricsToStream(session) {
            const sessionId = session?.session_id || session?.sessionId;
            if (!sessionId) {
                return null;
            }
            // Fallback: infer RTP port from RTCP port (typically RTCP = RTP + 1)
            // If session.port is missing but rtcp_port exists, derive RTP port
            const fallbackPort = session?.rtcp_port ? normalizePort(Number(session.rtcp_port) - 1) : null;
            const port = normalizePort(session?.port) ?? fallbackPort;
            return {
                session_id: sessionId,
                display_name: sessionId,
                source_host: session?.source_host || null,
                bind_address: session?.bind_address || null,
                port,
                sample_rate: session?.sample_rate || null,
                channels: session?.channels || null,
                payload_type: session?.payload_type ?? null,
                multicast: Boolean(session?.multicast),
                multicast_group: session?.multicast_group || null,
                status: 'active',
                existing_session: true,
                synthetic: true,
                auto_start: Boolean(session?.auto_start),
            };
        }

        /**
         * Merge metadata from source stream into target stream.
         * Only overwrites fields that are missing or invalid in target.
         * @param {Object} target - Target stream object (mutated in place)
         * @param {Object} source - Source stream object (read-only)
         */
        function mergeStreamMetadata(target, source) {
            if (!source) {
                return;
            }
            const fields = [
                'display_name',
                'source_host',
                'bind_address',
                'port',
                'sample_rate',
                'channels',
                'payload_type',
                'multicast',
                'multicast_group',
            ];
            fields.forEach((field) => {
                if (
                    (target[field] === undefined ||
                        target[field] === null ||
                        target[field] === '' ||
                        (typeof target[field] === 'number' && Number.isNaN(target[field]))) &&
                    source[field] !== undefined &&
                    source[field] !== null &&
                    source[field] !== ''
                ) {
                    target[field] = source[field];
                }
            });
        }

        /**
         * Merge active RTP sessions (from daemon telemetry) into scanned streams.
         * This ensures auto-started sessions appear in the UI without requiring a scan.
         *
         * Logic:
         * 1. Convert active sessions to synthetic streams
         * 2. Merge with existing scanned streams by session_id
         * 3. Remove synthetic entries that are no longer active
         * 4. Mark all active streams with existing_session=true
         */
        function mergeActiveSessionsIntoStreams() {
            const map = new Map();
            // Start with all scanned streams
            rtpState.streams.forEach((stream) => {
                map.set(stream.session_id, { ...stream });
            });

            const activeIds = new Set();

            // Process active sessions from daemon
            rtpState.activeSessions.forEach((session) => {
                const synthetic = sessionMetricsToStream(session);
                if (!synthetic) {
                    return;
                }
                activeIds.add(synthetic.session_id);
                if (map.has(synthetic.session_id)) {
                    // Merge metadata from telemetry into existing stream
                    const current = map.get(synthetic.session_id);
                    const merged = { ...current };
                    mergeStreamMetadata(merged, synthetic);
                    merged.status = 'active';
                    merged.existing_session = true;
                    merged.synthetic = current.synthetic || synthetic.synthetic;
                    merged.auto_start = merged.auto_start || synthetic.auto_start;
                    map.set(synthetic.session_id, merged);
                } else {
                    // Add new synthetic stream (auto-started, not yet scanned)
                    map.set(synthetic.session_id, synthetic);
                }
            });

            // Clean up: remove synthetic streams that are no longer active
            for (const [sessionId, stream] of map.entries()) {
                if (!activeIds.has(sessionId)) {
                    if (stream.synthetic) {
                        // Synthetic stream no longer active - remove from UI
                        map.delete(sessionId);
                        continue;
                    }
                    // Scanned stream not active - keep but mark as inactive
                    stream.existing_session = false;
                    stream.auto_start = false;
                }
            }

            rtpState.streams = Array.from(map.values());
        }

        async function fetchDevices() {
            try {
                const res = await fetch(API + '/devices');
                const data = await res.json();
                deviceList = data.devices;
                updateDeviceSelect();
            } catch (e) {
                console.error('Failed to fetch devices:', e);
            }
        }

        function updateDeviceSelect() {
            const select = document.getElementById('alsaDevice');
            select.innerHTML = '';

            deviceList.forEach(device => {
                const opt = document.createElement('option');
                opt.value = device.id;
                opt.textContent = device.name;
                if (device.id === currentAlsaDevice ||
                    currentAlsaDevice.includes(device.card)) {
                    opt.selected = true;
                }
                select.appendChild(opt);
            });

            // If current device not in list, add it
            const ids = deviceList.map(d => d.id);
            if (currentAlsaDevice && !ids.some(id => currentAlsaDevice.includes(id.split(',')[0].split('=')[1]))) {
                const opt = document.createElement('option');
                opt.value = currentAlsaDevice;
                opt.textContent = currentAlsaDevice;
                opt.selected = true;
                select.insertBefore(opt, select.firstChild);
            }
        }

        async function fetchStatus() {
            try {
                const res = await fetch(API + '/status');
                const data = await res.json();

                setStatus('daemonStatus', data.daemon_running ? 'Running' : 'Stopped', data.daemon_running);
                setStatus('eqStatus', data.eq_active ? 'ON' : 'OFF', data.eq_active);
                const inputMode = data.input_mode || (data.settings?.rtp_enabled ? 'rtp' : 'pipewire');
                lastPipewireConnected = data.pipewire_connected;
                updatePipewireCard(inputMode, lastPipewireConnected);
                setInputModeStatus(inputMode, lastPipewireConnected);

                if (currentAlsaDevice !== data.settings.alsa_device) {
                    currentAlsaDevice = data.settings.alsa_device;
                    updateDeviceSelect();
                }
            } catch (e) {
                console.error('Failed to fetch status:', e);
            }
        }

        function setStatus(id, text, ok) {
            const el = document.getElementById(id);
            el.querySelector('.value').textContent = text;
            el.classList.remove('ok', 'error');
            el.classList.add(ok ? 'ok' : 'error');
        }

        let inputModeMessageTimeout = null;

        function updateInputModeOptionStyles() {
            inputModeOptions.forEach(option => {
                const mode = option.dataset.mode;
                const radio = option.querySelector('input[type="radio"]');
                const isActive = mode === inputModeState.current;
                option.classList.toggle('active', isActive);
                option.classList.toggle('disabled', inputModeState.switching);
                if (radio) {
                    radio.checked = isActive;
                    radio.disabled = inputModeState.switching;
                }
            });
        }

        function showInputModeMessage(text, success) {
            if (inputModeMessageTimeout) {
                clearTimeout(inputModeMessageTimeout);
            }
            inputModeMessage.textContent = text || '';
            inputModeMessage.classList.remove('success', 'error');
            if (text) {
                inputModeMessage.classList.add(success ? 'success' : 'error');
                inputModeMessageTimeout = setTimeout(() => {
                    inputModeMessage.classList.remove('success', 'error');
                }, 4000);
            }
        }

        function setInputModeStatus(mode, pipewireConnected) {
            const normalized = mode === 'rtp' ? 'rtp' : 'pipewire';
            if (!inputModeState.switching) {
                inputModeState.current = normalized;
            }
            const description = normalized === 'rtp'
                ? '現在: RTP (ネットワーク入力)'
                : `現在: PipeWire (${pipewireConnected ? '接続済み' : '未接続'})`;
            inputModeStatusText.textContent = description;
            updateInputModeOptionStyles();
        }

        function updatePipewireCard(normalizedMode, pipewireConnected) {
            if (normalizedMode === 'rtp') {
                pwStatusLabel.textContent = 'Input Mode';
                setStatus('pwStatus', 'RTP Active', true);
            } else {
                pwStatusLabel.textContent = 'PipeWire';
                setStatus('pwStatus', pipewireConnected ? 'OK' : 'N/A', pipewireConnected);
            }
        }

        async function requestInputModeChange(mode) {
            if (!mode || mode === inputModeState.current || inputModeState.switching) {
                return;
            }
            inputModeState.switching = true;
            updateInputModeOptionStyles();
            inputModeRestartHint.textContent = 'デーモンを再起動しています...';
            showInputModeMessage('モード切り替え中...', true);
            try {
                const res = await fetch(INPUT_MODE_API + '/switch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode }),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok || !data.success) {
                    throw new Error(data.detail || data.message || '切り替えに失敗しました');
                }
                inputModeState.current = data.current_mode || mode;
                showInputModeMessage(data.message || 'モードを切り替えました', true);
            } catch (e) {
                showInputModeMessage('エラー: ' + e.message, false);
            } finally {
                inputModeState.switching = false;
                inputModeRestartHint.textContent = '切り替え時にデーモンを自動再起動します';
                updateInputModeOptionStyles();
                fetchStatus();
            }
        }

        function showMessage(text, success) {
            const el = document.getElementById('settingsMessage');
            el.textContent = text;
            el.classList.remove('success', 'error');
            el.classList.add(success ? 'success' : 'error');
            setTimeout(() => el.classList.remove('success', 'error'), 4000);
        }

        function showRtpMessage(text, success) {
            rtpMessage.textContent = text;
            rtpMessage.classList.remove('success', 'error');
            rtpMessage.classList.add(success ? 'success' : 'error');
            setTimeout(() => rtpMessage.classList.remove('success', 'error'), 4000);
        }

        function updateRtpScanStatus() {
            if (rtpState.scanning) {
                rtpScanStatus.textContent = 'スキャン中...';
                return;
            }
            if (rtpState.lastScanAt) {
                const updated = new Date(rtpState.lastScanAt);
                const hh = String(updated.getHours()).padStart(2, '0');
                const mm = String(updated.getMinutes()).padStart(2, '0');
                rtpScanStatus.textContent = `${hh}:${mm} に更新 (${rtpState.streams.length}件)`;
            } else {
                rtpScanStatus.textContent = '未スキャン';
            }
        }

        function renderRtpOptions() {
            rtpSelect.innerHTML = '';
            if (!rtpState.streams.length) {
                const placeholder = document.createElement('option');
                placeholder.value = '';
                placeholder.textContent = rtpState.scanning ? 'スキャン中...' : '候補がありません';
                rtpSelect.appendChild(placeholder);
                rtpSelect.disabled = true;
                rtpStartBtn.disabled = true;
                updateRtpDetails();
                return;
            }
            rtpSelect.disabled = false;
            rtpState.streams.forEach((stream) => {
                const opt = document.createElement('option');
                opt.value = stream.session_id;
                const hostLabel = stream.source_host || stream.bind_address || '-';
                const portLabel = stream.port ?? '-';
                const badges = [];
                if (stream.existing_session) {
                    badges.push('稼働中');
                }
                if (stream.auto_start) {
                    badges.push('自動起動');
                }
                const suffix = badges.length ? ` (${badges.join(' / ')})` : '';
                opt.textContent = `${stream.display_name} • ${hostLabel}:${portLabel}${suffix}`;
                rtpSelect.appendChild(opt);
            });
            if (!rtpState.selectedId || !rtpState.streams.some((s) => s.session_id === rtpState.selectedId)) {
                rtpState.selectedId = rtpState.streams[0].session_id;
            }
            rtpSelect.value = rtpState.selectedId;
            const selectedStream = rtpState.streams.find((item) => item.session_id === rtpState.selectedId);
            rtpStartBtn.disabled = !selectedStream || (selectedStream.synthetic && selectedStream.existing_session);
            updateRtpDetails();
        }

        function updateRtpDetails() {
            const stream = rtpState.streams.find((item) => item.session_id === rtpState.selectedId);
            if (!stream) {
                rtpDetails.textContent = 'スキャンして候補を読み込みます。';
                return;
            }
            const tags = [`<span class="pill">${stream.status || 'unknown'}</span>`];
            if (stream.existing_session) {
                tags.push('<span class="pill pill-active">稼働中</span>');
            }
            if (stream.auto_start) {
                tags.push('<span class="pill pill-auto">自動起動</span>');
            }
            const infoBits = [];
            if (stream.sample_rate) infoBits.push(`${stream.sample_rate} Hz`);
            if (stream.channels) infoBits.push(`${stream.channels}ch`);
            if (stream.payload_type !== null && stream.payload_type !== undefined) {
                infoBits.push(`PT${stream.payload_type}`);
            }
            const listenHost = stream.bind_address || stream.source_host || '-';
            const listenPort = stream.port ?? '-';
            const sourceHost = stream.source_host || '-';
            rtpDetails.innerHTML = `
                <div><strong>${stream.display_name}</strong></div>
                <div>受信: <strong>${listenHost}</strong>${listenPort !== '-' ? ':' + listenPort : ''}</div>
                <div>ソース: <strong>${sourceHost}</strong></div>
                <div>${tags.join(' ')}</div>
                <div>${infoBits.length ? infoBits.join(' / ') : '詳細情報なし'}</div>
                ${stream.synthetic ? '<div class="rtp-note">自動起動済みのセッションを監視中です。停止ボタンで制御できます。</div>' : ''}
            `;
        }

        function updateActiveSessionText() {
            if (!rtpState.activeSessions.length) {
                rtpActiveSession.textContent = '稼働中のRTPセッション: なし';
                rtpStopBtn.disabled = true;
                delete rtpStopBtn.dataset.targetId;
                return;
            }
            const labels = rtpState.activeSessions
                .map((session) => {
                    const auto = session.auto_start ? '（自動起動）' : '';
                    return `${session.session_id}${auto}`;
                })
                .join(', ');
            rtpActiveSession.innerHTML = `稼働中のRTPセッション: <span>${labels}</span>`;
            rtpStopBtn.disabled = false;
            rtpStopBtn.dataset.targetId = rtpState.activeSessions[0].session_id;
        }

        async function refreshRtpSessions() {
            try {
                const res = await fetch(RTP_API + '/sessions');
                if (!res.ok) {
                    return;
                }
                const data = await res.json();
                rtpState.activeSessions = Array.isArray(data.sessions) ? data.sessions : [];
                mergeActiveSessionsIntoStreams();
                renderRtpOptions();
                updateActiveSessionText();
            } catch (e) {
                console.error('Failed to refresh RTP sessions:', e);
            }
        }

        function buildRtpSessionPayload(stream) {
            const payload = {
                session_id: stream.session_id,
                endpoint: {
                    bind_address: stream.bind_address || '0.0.0.0',
                    port: stream.port,
                },
                format: {
                    sample_rate: stream.sample_rate || 48000,
                    channels: stream.channels || 2,
                    payload_type: (stream.payload_type !== undefined && stream.payload_type !== null) ? stream.payload_type : 97,
                },
            };
            if (stream.source_host) {
                payload.endpoint.source_host = stream.source_host;
            }
            if (stream.multicast) {
                payload.endpoint.multicast = true;
            }
            if (stream.multicast_group) {
                payload.endpoint.multicast_group = stream.multicast_group;
            }
            return payload;
        }

        async function scanRtpStreams() {
            if (rtpState.scanning) return;
            rtpState.scanning = true;
            rtpScanBtn.disabled = true;
            rtpScanBtn.textContent = 'スキャン中...';
            updateRtpScanStatus();
            try {
                const res = await fetch(RTP_API + '/discover');
                const data = await res.json().catch(() => ({}));
                if (!res.ok) {
                    throw new Error(data.detail || data.message || 'スキャンに失敗しました');
                }
                rtpState.streams = (data.streams || []).map((stream) => ({
                    ...stream,
                    synthetic: Boolean(stream.synthetic),
                }));
                rtpState.selectedId = rtpState.streams[0]?.session_id || '';
                rtpState.lastScanAt = data.scanned_at_unix_ms || Date.now();
                renderRtpOptions();
                updateRtpScanStatus();
                showRtpMessage(`候補を更新しました (${rtpState.streams.length}件)`, true);
                await refreshRtpSessions();
            } catch (e) {
                console.error('RTP scan failed:', e);
                showRtpMessage('エラー: ' + e.message, false);
            } finally {
                rtpState.scanning = false;
                rtpScanBtn.disabled = false;
                rtpScanBtn.textContent = 'RTP入力をスキャン';
                updateRtpScanStatus();
            }
        }

        async function startSelectedRtpSession() {
            const stream = rtpState.streams.find((item) => item.session_id === rtpState.selectedId);
            if (!stream) {
                showRtpMessage('候補を選択してください', false);
                return;
            }
            const payload = buildRtpSessionPayload(stream);
            rtpStartBtn.disabled = true;
            rtpStartBtn.textContent = '開始中...';
            try {
                const res = await fetch(RTP_API + '/sessions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok) {
                    throw new Error(data.detail || data.message || '開始に失敗しました');
                }
                showRtpMessage(`${stream.display_name} を開始しました`, true);
                await refreshRtpSessions();
            } catch (e) {
                console.error('Failed to start RTP session:', e);
                showRtpMessage('エラー: ' + e.message, false);
            } finally {
                rtpStartBtn.disabled = false;
                rtpStartBtn.textContent = '開始';
            }
        }

        async function stopActiveRtpSession() {
            const targetId = rtpStopBtn.dataset.targetId;
            if (!targetId) {
                showRtpMessage('停止できるセッションがありません', false);
                return;
            }
            rtpStopBtn.disabled = true;
            rtpStopBtn.textContent = '停止中...';
            try {
                const res = await fetch(`${RTP_API}/sessions/${encodeURIComponent(targetId)}`, {
                    method: 'DELETE',
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok) {
                    throw new Error(data.detail || data.message || '停止に失敗しました');
                }
                showRtpMessage(data.message || 'RTPを停止しました', true);
                await refreshRtpSessions();
            } catch (e) {
                console.error('Failed to stop RTP session:', e);
                showRtpMessage('エラー: ' + e.message, false);
            } finally {
                rtpStopBtn.disabled = false;
                rtpStopBtn.textContent = '停止';
            }
        }

        document.getElementById('settingsForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const newDevice = document.getElementById('alsaDevice').value;
            const btn = document.getElementById('saveBtn');

            btn.disabled = true;
            btn.textContent = 'Saving...';

            try {
                const res = await fetch(API + '/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ alsa_device: newDevice }),
                });
                const data = await res.json();

                if (data.success && data.restart_required) {
                    btn.textContent = 'Restarting...';
                    await fetch(API + '/restart', { method: 'POST' });
                    showMessage('Daemon restarting...', true);
                    setTimeout(fetchStatus, 2000);
                } else {
                    showMessage(data.message, data.success);
                }
            } catch (e) {
                showMessage('Error: ' + e.message, false);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Save & Restart';
            }
        });

        inputModeRadios.forEach(radio => {
            radio.addEventListener('change', (event) => {
                const mode = event.target.value;
                requestInputModeChange(mode);
            });
        });

        rtpScanBtn.addEventListener('click', scanRtpStreams);
        rtpStartBtn.addEventListener('click', startSelectedRtpSession);
        rtpStopBtn.addEventListener('click', stopActiveRtpSession);
        rtpSelect.addEventListener('change', (event) => {
            rtpState.selectedId = event.target.value;
            const selectedStream = rtpState.streams.find((item) => item.session_id === rtpState.selectedId);
            rtpStartBtn.disabled = !selectedStream || (selectedStream.synthetic && selectedStream.existing_session);
            updateRtpDetails();
        });

        // OPRA Functions
        let selectedProduct = null;
        let searchTimeout = null;

        function showOpraMessage(text, success) {
            const el = document.getElementById('opraMessage');
            el.textContent = text;
            el.classList.remove('success', 'error');
            el.classList.add(success ? 'success' : 'error');
            setTimeout(() => el.classList.remove('success', 'error'), 4000);
        }

        async function searchOpra(query) {
            if (!query || query.length < 2) {
                document.getElementById('opraResults').innerHTML = '';
                return;
            }
            try {
                const res = await fetch(API + '/opra/search?q=' + encodeURIComponent(query) + '&limit=20');
                if (!res.ok) {
                    const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }));
                    const container = document.getElementById('opraResults');

                    if (res.status === 503) {
                        // Service unavailable - database not initialized
                        container.innerHTML = '<div style="color:#e74c3c; font-size:12px; padding:8px; line-height:1.5;">' +
                            '<strong>Headphone database is not available</strong><br>' +
                            'Please contact the administrator to initialize the database.' +
                            '</div>';
                    } else {
                        // Other errors - safely display message without XSS risk
                        const errorDiv = document.createElement('div');
                        errorDiv.style.cssText = 'color:#e74c3c; font-size:12px; padding:8px;';
                        errorDiv.textContent = 'Error: ' + (errorData.detail || 'Search failed');
                        container.innerHTML = '';
                        container.appendChild(errorDiv);
                    }
                    console.error('OPRA search failed:', errorData);
                    return;
                }
                const data = await res.json();
                renderOpraResults(data.results);
            } catch (e) {
                console.error('OPRA search failed:', e);
                const container = document.getElementById('opraResults');
                container.innerHTML = '<div style="color:#e74c3c; font-size:12px; padding:8px;">Network error. Please check the server.</div>';
            }
        }

        function renderOpraResults(results) {
            const container = document.getElementById('opraResults');
            if (!results.length) {
                container.innerHTML = '<div style="color:#666; font-size:12px; padding:8px;">No results</div>';
                return;
            }
            container.innerHTML = results.map(r => `
                <div class="opra-item" data-id="${r.id}" style="padding:8px; cursor:pointer; border-bottom:1px solid #0f3460;">
                    <div style="font-weight:500;">${r.vendor.name} ${r.name}</div>
                    <div style="font-size:11px; color:#666;">${r.eq_profiles.length} EQ profile(s)</div>
                </div>
            `).join('');

            // Add click handlers
            container.querySelectorAll('.opra-item').forEach(el => {
                el.addEventListener('click', () => selectProduct(results.find(r => r.id === el.dataset.id)));
                el.addEventListener('mouseenter', () => el.style.background = '#0f3460');
                el.addEventListener('mouseleave', () => el.style.background = '');
            });
        }

        function selectProduct(product) {
            selectedProduct = product;
            document.getElementById('opraResults').innerHTML = '';
            document.getElementById('opraSearch').value = '';
            document.getElementById('opraSelected').style.display = 'block';
            document.getElementById('selectedName').textContent = product.name;
            document.getElementById('selectedVendor').textContent = product.vendor.name;

            // Populate EQ profiles
            const select = document.getElementById('opraEqSelect');
            select.innerHTML = product.eq_profiles.map(eq =>
                `<option value="${eq.id}">${eq.author || 'unknown'} - ${eq.details || 'EQ'}</option>`
            ).join('');

            document.getElementById('applyOpraBtn').disabled = false;
        }

        document.getElementById('opraSearch').addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => searchOpra(e.target.value), 300);
        });

        document.getElementById('applyOpraBtn').addEventListener('click', async () => {
            const eqId = document.getElementById('opraEqSelect').value;
            if (!eqId) {
                showOpraMessage('Select an EQ profile', false);
                return;
            }
            const btn = document.getElementById('applyOpraBtn');
            btn.disabled = true;
            btn.textContent = 'Applying...';
            try {
                const applyCorrection = document.getElementById('modernTargetCheckbox').checked;
                const res = await fetch(API + '/opra/apply/' + encodeURIComponent(eqId) + '?apply_correction=' + applyCorrection, { method: 'POST' });
                const data = await res.json();
                showOpraMessage(data.message, data.success);
                if (data.success && data.restart_required) {
                    btn.textContent = 'Restarting...';
                    await fetch(API + '/restart', { method: 'POST' });
                    setTimeout(fetchStatus, 2000);
                }
            } catch (e) {
                showOpraMessage('Error: ' + e.message, false);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Apply EQ';
            }
        });

        document.getElementById('deactivateEqBtn').addEventListener('click', async () => {
            try {
                const res = await fetch(API + '/eq/deactivate', { method: 'POST' });
                const data = await res.json();
                showOpraMessage(data.message, data.success);
                if (data.success && data.restart_required) {
                    await fetch(API + '/restart', { method: 'POST' });
                    setTimeout(fetchStatus, 2000);
                }
            } catch (e) {
                showOpraMessage('Error: ' + e.message, false);
            }
        });

        // Phase Type Functions
        let currentPhaseType = 'minimum';
        let isLowLatencyModeEnabled = false;
        const HYBRID_PHASE_NOTE = 'ハイブリッドは100Hz以下を最小位相、100Hz以上を線形位相（10ms整列）で処理します。';
        const HYBRID_PHASE_WARNING_TEXT = HYBRID_PHASE_NOTE + ' 約10msの整列ディレイが発生し、低遅延モードとは併用できません。';

        function updatePhaseOptionAvailability() {
            const hybridOption = document.querySelector('#phaseType option[value="hybrid"]');
            if (!hybridOption) return;
            hybridOption.disabled = isLowLatencyModeEnabled;
            hybridOption.title = isLowLatencyModeEnabled ? '低遅延モード中は選択できません' : '100Hz以下は最小位相 / 100Hz以上は線形位相（10ms整列）';
        }

        async function fetchPartitionStatus() {
            try {
                const res = await fetch(API + '/partitioned-convolution');
                if (!res.ok) return;
                const data = await res.json();
                const previouslyEnabled = isLowLatencyModeEnabled;
                isLowLatencyModeEnabled = Boolean(data.enabled);
                updatePhaseOptionAvailability();
                if (isLowLatencyModeEnabled && currentPhaseType !== 'minimum') {
                    currentPhaseType = 'minimum';
                    const select = document.getElementById('phaseType');
                    select.value = 'minimum';
                    updatePhaseWarning('minimum');
                } else if (!isLowLatencyModeEnabled && previouslyEnabled) {
                    updatePhaseWarning(currentPhaseType);
                }
            } catch (e) {
                console.error('Failed to fetch partitioned-convolution status:', e);
            }
        }

        async function fetchPhaseType() {
            const select = document.getElementById('phaseType');
            const warning = document.getElementById('phaseWarning');
            try {
                const res = await fetch(API + '/daemon/phase-type');
                if (!res.ok) {
                    // Daemon not running or error - disable UI and show message
                    select.disabled = true;
                    warning.classList.remove('visible');
                    try {
                        const err = await res.json();
                        // RFC 9457: use detail field for error message
                        showPhaseMessage(err.detail || 'Phase type unavailable', false);
                    } catch {
                        showPhaseMessage('Phase type unavailable (HTTP ' + res.status + ')', false);
                    }
                    return;
                }
                const data = await res.json();
                currentPhaseType = data.phase_type;
                select.value = data.phase_type;
                select.disabled = false;
                // Use API's latency_warning if provided
                updatePhaseWarning(data.phase_type, data.latency_warning);
                updatePhaseOptionAvailability();
            } catch (e) {
                console.error('Failed to fetch phase type:', e);
                select.disabled = true;
                warning.classList.remove('visible');
            }
        }

        function updatePhaseWarning(phaseType, apiWarning) {
            const warning = document.getElementById('phaseWarning');
            const message = apiWarning || HYBRID_PHASE_WARNING_TEXT;
            if (phaseType === 'hybrid') {
                warning.textContent = '⚠️ ' + message;
                warning.classList.add('visible');
                return;
            }
            warning.classList.remove('visible');
        }

        function showPhaseMessage(text, success) {
            const el = document.getElementById('phaseMessage');
            el.textContent = text;
            el.classList.remove('success', 'error');
            el.classList.add(success ? 'success' : 'error');
            setTimeout(() => el.classList.remove('success', 'error'), 4000);
        }

        document.getElementById('phaseType').addEventListener('change', async (e) => {
            const newPhaseType = e.target.value;
            updatePhaseWarning(newPhaseType);

            if (newPhaseType === currentPhaseType) return;

            const select = e.target;
            select.disabled = true;

            try {
                const res = await fetch(API + '/daemon/phase-type', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ phase_type: newPhaseType }),
                });
                const data = await res.json();

                if (res.ok && data.success) {
                    currentPhaseType = newPhaseType;
                    let successMessage = 'Phase type updated to ' + newPhaseType;
                    if (data.data && data.data.partition_disabled) {
                        isLowLatencyModeEnabled = false;
                        updatePhaseOptionAvailability();
                        successMessage += '（ハイブリッドに切り替えたため低遅延モードを自動で無効化しました）';
                    }
                    showPhaseMessage(successMessage, true);
                    // Refresh partition status to keep availability in sync
                    await fetchPartitionStatus();
                } else {
                    // Revert selection
                    select.value = currentPhaseType;
                    updatePhaseWarning(currentPhaseType);
                    // RFC 9457: error response has 'detail' field
                    showPhaseMessage(data.detail || data.message || 'Failed to update phase type', false);
                }
            } catch (e) {
                select.value = currentPhaseType;
                updatePhaseWarning(currentPhaseType);
                showPhaseMessage('Error: ' + e.message, false);
            } finally {
                select.disabled = false;
            }
        });

        // Crossfeed Functions
        const crossfeedState = {
            enabled: false,
            headSize: 'm',
            isApplying: false
        };

        function showCrossfeedMessage(text, success) {
            const el = document.getElementById('crossfeedMessage');
            el.textContent = text;
            el.classList.remove('success', 'error');
            el.classList.add(success ? 'success' : 'error');
            setTimeout(() => el.classList.remove('success', 'error'), 4000);
        }

        function updateCrossfeedToggle(enabled) {
            const toggle = document.getElementById('crossfeedToggle');
            crossfeedState.enabled = enabled;
            if (enabled) {
                toggle.classList.add('active');
            } else {
                toggle.classList.remove('active');
            }
            updateCrossfeedStatusDisplay();
        }

        function updateHeadSizeButtons(headSize) {
            document.querySelectorAll('.head-size-btn').forEach(btn => {
                if (btn.dataset.size === headSize) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
            crossfeedState.headSize = headSize;
        }

        function updateCrossfeedStatusDisplay() {
            const statusEl = document.getElementById('crossfeedStatus');
            const indicatorEl = document.getElementById('crossfeedStatusIndicator');
            const statusTextEl = document.getElementById('crossfeedStatusText');

            if (crossfeedState.enabled) {
                statusEl.style.display = 'block';
                indicatorEl.classList.add('active');
                statusTextEl.textContent = '有効';
            } else {
                statusEl.style.display = 'none';
                indicatorEl.classList.remove('active');
                statusTextEl.textContent = '無効';
            }
        }

        function setCrossfeedLoading(loading) {
            const loadingEl = document.getElementById('crossfeedLoading');
            const toggle = document.getElementById('crossfeedToggle');
            const headSizeBtns = document.querySelectorAll('.head-size-btn');

            crossfeedState.isApplying = loading;
            if (loading) {
                loadingEl.classList.add('visible');
                toggle.style.pointerEvents = 'none';
                headSizeBtns.forEach(btn => btn.disabled = true);
            } else {
                loadingEl.classList.remove('visible');
                toggle.style.pointerEvents = 'auto';
                headSizeBtns.forEach(btn => btn.disabled = false);
            }
        }

        async function fetchCrossfeedStatus() {
            try {
                const res = await fetch(API + '/crossfeed/status');
                if (!res.ok) {
                    console.error('Failed to fetch crossfeed status:', res.status);
                    return;
                }
                const data = await res.json();
                updateCrossfeedToggle(data.enabled || false);
                if (data.headSize) {
                    updateHeadSizeButtons(data.headSize.toLowerCase());
                }
            } catch (e) {
                console.error('Failed to fetch crossfeed status:', e);
            }
        }

        async function toggleCrossfeed() {
            if (crossfeedState.isApplying) return;

            const wasEnabled = crossfeedState.enabled;
            setCrossfeedLoading(true);
            try {
                const endpoint = wasEnabled ? '/crossfeed/disable' : '/crossfeed/enable';
                const res = await fetch(API + endpoint, { method: 'POST' });
                const data = await res.json();

                if (res.ok && data.success) {
                    await fetchCrossfeedStatus();
                    // Use API message if available, otherwise use state-based message
                    const message = data.message || (wasEnabled ? 'クロスフィードを無効化しました' : 'クロスフィードを有効化しました');
                    showCrossfeedMessage(message, true);
                } else {
                    showCrossfeedMessage(data.detail || data.message || 'クロスフィードの切り替えに失敗しました', false);
                }
            } catch (e) {
                showCrossfeedMessage('エラー: ' + e.message, false);
            } finally {
                setCrossfeedLoading(false);
            }
        }

        async function setHeadSize(size) {
            if (crossfeedState.isApplying || crossfeedState.headSize === size) return;

            setCrossfeedLoading(true);
            try {
                const res = await fetch(API + '/crossfeed/size/' + encodeURIComponent(size), { method: 'POST' });
                const data = await res.json();

                if (res.ok && data.success) {
                    updateHeadSizeButtons(data.headSize || size);
                    showCrossfeedMessage('頭サイズを ' + size.toUpperCase() + ' に変更しました', true);
                } else {
                    showCrossfeedMessage(data.detail || data.message || '頭サイズの変更に失敗しました', false);
                }
            } catch (e) {
                showCrossfeedMessage('エラー: ' + e.message, false);
            } finally {
                setCrossfeedLoading(false);
            }
        }

        // Crossfeed Event Listeners
        document.getElementById('crossfeedToggle').addEventListener('click', toggleCrossfeed);
        document.querySelectorAll('.head-size-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const size = btn.dataset.size;
                setHeadSize(size);
            });
        });

        // Initial load
        setInputModeStatus('pipewire', false);
        fetchDevices();
        fetchStatus();
        fetchPhaseType();
        fetchPartitionStatus();
        fetchCrossfeedStatus();
        updateRtpScanStatus();
        refreshRtpSessions();
        setInterval(fetchStatus, 5000);
        setInterval(fetchPhaseType, 5000);
        setInterval(fetchPartitionStatus, 5000);
        setInterval(fetchCrossfeedStatus, 5000);
        setInterval(refreshRtpSessions, 7000);
    </script>
</body>
</html>
"""
