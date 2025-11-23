"""Admin dashboard HTML template."""


def get_admin_html() -> str:
    """Return admin dashboard HTML"""
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Upsampler - Admin</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
            max-width: 600px;
            margin: 0 auto;
        }
        h1 { color: #ff6b6b; margin-bottom: 8px; font-size: 1.4em; }
        .subtitle { color: #666; font-size: 12px; margin-bottom: 20px; }
        h2 { color: #888; font-size: 11px; margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 1px; }
        .card {
            background: #16213e;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
        .stat-item {
            background: #0f3460;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-item .label { font-size: 10px; color: #666; text-transform: uppercase; }
        .stat-item .value { font-size: 18px; font-weight: 600; margin-top: 4px; color: #00d4ff; }
        .stat-item.warning .value { color: #ffaa00; }
        .stat-item.error .value { color: #ff4444; }
        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .status-item { text-align: center; flex: 1; }
        .status-item .label { font-size: 10px; color: #666; text-transform: uppercase; }
        .status-item .value { font-size: 14px; font-weight: 600; margin-top: 2px; }
        .status-item.ok .value { color: #00ff88; }
        .status-item.error .value { color: #ff4444; }
        .btn-row { display: flex; gap: 10px; margin-top: 12px; }
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
        .btn-success { background: #00ff88; color: #000; }
        .btn-success:hover { background: #00cc6a; }
        .btn-danger { background: #ff4444; color: #fff; }
        .btn-danger:hover { background: #cc3333; }
        .btn-warning { background: #ffaa00; color: #000; }
        .btn-warning:hover { background: #cc8800; }
        button:disabled { background: #555; color: #888; cursor: not-allowed; }
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #0f3460;
            font-size: 13px;
        }
        .info-row:last-child { border-bottom: none; }
        .info-row .label { color: #888; }
        .info-row .value { color: #eee; font-family: monospace; }
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
        .back-link { color: #00d4ff; text-decoration: none; font-size: 13px; }
        .back-link:hover { text-decoration: underline; }
        .eq-section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .eq-profile-name {
            font-size: 14px;
            font-weight: 600;
            color: #00d4ff;
        }
        .eq-inactive { color: #666; }
        .copy-btn {
            background: #0f3460;
            border: none;
            color: #00d4ff;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        .copy-btn:hover { background: #1a4b7c; }
        .copy-btn.copied { background: #00ff88; color: #000; }
        .eq-section-label {
            font-size: 10px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin: 12px 0 6px;
            padding-bottom: 4px;
            border-bottom: 1px solid #0f3460;
        }
        .eq-section-label:first-of-type { margin-top: 0; }
        .eq-filters {
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            font-size: 11px;
            line-height: 1.6;
            color: #ccc;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .eq-attribution {
            font-size: 10px;
            color: #666;
            margin-top: 8px;
        }
        .eq-attribution a { color: #00d4ff; }
        /* Custom EQ Upload styles */
        .upload-zone {
            border: 2px dashed #0f3460;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            transition: border-color 0.2s, background 0.2s;
            cursor: pointer;
        }
        .upload-zone:hover, .upload-zone.dragover {
            border-color: #00d4ff;
            background: #0f346020;
        }
        .upload-zone input[type="file"] { display: none; }
        .upload-zone .icon { font-size: 32px; margin-bottom: 8px; }
        .upload-zone .text { font-size: 13px; color: #888; }
        .upload-zone .text strong { color: #00d4ff; }
        .upload-info {
            margin-top: 12px;
            padding: 10px;
            background: #0f3460;
            border-radius: 6px;
            font-size: 12px;
        }
        .upload-info.error { background: #ff444440; }
        .upload-info.warning { background: #ffaa0040; }
        .upload-info .filename { font-weight: 600; color: #00d4ff; }
        .upload-info .details { color: #888; margin-top: 4px; }
        .upload-info .errors { color: #ff6b6b; margin-top: 6px; }
        .upload-info .warnings { color: #ffaa00; margin-top: 4px; }
        .profile-list { margin-top: 12px; }
        .profile-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 12px;
            background: #0f3460;
            border-radius: 6px;
            margin-bottom: 8px;
        }
        .profile-item:last-child { margin-bottom: 0; }
        .profile-item .info { flex: 1; }
        .profile-item .name { font-weight: 500; font-size: 13px; }
        .profile-item .meta { font-size: 11px; color: #666; margin-top: 2px; }
        .profile-item .type-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 9px;
            text-transform: uppercase;
            margin-left: 6px;
        }
        .profile-item .type-badge.opra { background: #00d4ff30; color: #00d4ff; }
        .profile-item .type-badge.custom { background: #ffaa0030; color: #ffaa00; }
        .profile-item .actions { display: flex; gap: 6px; }
        .profile-item .actions button {
            flex: none;
            padding: 6px 10px;
            font-size: 11px;
        }
        .btn-small { padding: 6px 12px !important; font-size: 12px !important; }
        .btn-secondary { background: #0f3460; color: #eee; }
        .btn-secondary:hover { background: #1a4b7c; }
        .empty-state { color: #666; font-size: 13px; text-align: center; padding: 20px; }
    </style>
</head>
<body>
    <h1>GPU Upsampler Admin</h1>
    <div class="subtitle"><a href="/" class="back-link">&larr; Back to User Page</a></div>

    <h2>Daemon Control</h2>
    <div class="card">
        <div class="status-row">
            <div class="status-item" id="daemonStatus">
                <div class="label">Status</div>
                <div class="value">-</div>
            </div>
            <div class="status-item" id="pidStatus">
                <div class="label">PID</div>
                <div class="value">-</div>
            </div>
            <div class="status-item" id="pwStatus">
                <div class="label">PipeWire</div>
                <div class="value">-</div>
            </div>
        </div>
        <div class="btn-row">
            <button class="btn-success" id="startBtn">Start</button>
            <button class="btn-danger" id="stopBtn">Stop</button>
            <button class="btn-warning" id="restartBtn">Restart</button>
        </div>
        <div id="controlMessage" class="message"></div>
    </div>

    <h2>Statistics</h2>
    <div class="card">
        <div class="stat-grid">
            <div class="stat-item" id="clipRate">
                <div class="label">Clip Rate</div>
                <div class="value">-</div>
            </div>
            <div class="stat-item" id="clipCount">
                <div class="label">Clipped Samples</div>
                <div class="value">-</div>
            </div>
            <div class="stat-item" id="totalSamples">
                <div class="label">Total Samples</div>
                <div class="value">-</div>
            </div>
            <div class="stat-item" id="eqStatus">
                <div class="label">EQ</div>
                <div class="value">-</div>
            </div>
        </div>
    </div>

    <h2>Sampling Rate</h2>
    <div class="card">
        <div class="stat-grid">
            <div class="stat-item" id="inputRate">
                <div class="label">Input</div>
                <div class="value">-</div>
            </div>
            <div class="stat-item" id="outputRate">
                <div class="label">Output</div>
                <div class="value">-</div>
            </div>
        </div>
    </div>

    <h2>System Info</h2>
    <div class="card">
        <div class="info-row">
            <span class="label">PID File</span>
            <span class="value" id="pidFile">-</span>
        </div>
        <div class="info-row">
            <span class="label">Binary</span>
            <span class="value" id="binaryPath">-</span>
        </div>
        <div class="info-row">
            <span class="label">ALSA Device</span>
            <span class="value" id="alsaDevice">-</span>
        </div>
        <div class="info-row">
            <span class="label">Upsample Ratio</span>
            <span class="value" id="upsampleRatio">-</span>
        </div>
    </div>

    <h2>適用中EQプロファイル</h2>
    <div class="card" id="eqProfileCard">
        <div class="eq-section-header">
            <span class="eq-profile-name" id="eqProfileName">-</span>
            <button class="copy-btn" id="copyEqBtn" style="display:none;">Copy</button>
        </div>
        <div id="eqContent">
            <div class="eq-inactive">EQ無効</div>
        </div>
    </div>

    <h2>カスタムEQアップロード</h2>
    <div class="card">
        <div class="upload-zone" id="uploadZone">
            <input type="file" id="eqFileInput" accept=".txt">
            <div class="icon">📁</div>
            <div class="text">
                <strong>ファイルを選択</strong> または ドラッグ&ドロップ<br>
                <span style="font-size:11px;">Equalizer APO形式 (.txt), 最大1MB</span>
            </div>
        </div>
        <div id="uploadInfo" style="display:none;"></div>
        <div class="btn-row" id="uploadActions" style="display:none;">
            <button class="btn-success btn-small" id="uploadBtn" disabled>アップロード</button>
            <button class="btn-secondary btn-small" id="cancelUploadBtn">キャンセル</button>
        </div>
        <div id="uploadMessage" class="message"></div>
    </div>

    <h2>保存済みEQプロファイル</h2>
    <div class="card">
        <div id="profileList" class="profile-list">
            <div class="empty-state">読み込み中...</div>
        </div>
    </div>

    <script>
        const API = '';
        let statsWebSocket = null;
        let wsReconnectAttempts = 0;
        const MAX_RECONNECT_ATTEMPTS = 5;

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = protocol + '//' + window.location.host + '/ws/stats';

            try {
                statsWebSocket = new WebSocket(wsUrl);

                statsWebSocket.onopen = () => {
                    console.log('WebSocket connected');
                    wsReconnectAttempts = 0;
                };

                statsWebSocket.onmessage = (event) => {
                    try {
                        const stats = JSON.parse(event.data);
                        updateStatsFromWebSocket(stats);
                    } catch (e) {
                        console.error('Failed to parse WebSocket message:', e);
                    }
                };

                statsWebSocket.onclose = () => {
                    console.log('WebSocket disconnected');
                    statsWebSocket = null;
                    // Attempt to reconnect
                    if (wsReconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                        wsReconnectAttempts++;
                        setTimeout(connectWebSocket, 2000 * wsReconnectAttempts);
                    }
                };

                statsWebSocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
            } catch (e) {
                console.error('Failed to create WebSocket:', e);
            }
        }

        function updateStatsFromWebSocket(stats) {
            // Stats - clip_rate is now a ratio (0-1), multiply by 100 for percentage
            const clipPct = (stats.clip_rate * 100).toFixed(4);
            setStat('clipRate', clipPct + '%', stats.clip_rate < 0.001 ? '' : (stats.clip_rate < 0.01 ? 'warning' : 'error'));
            setStat('clipCount', formatNumber(stats.clip_count), '');
            setStat('totalSamples', formatNumber(stats.total_samples), '');

            // Sample rates
            setStat('inputRate', formatSampleRate(stats.input_rate || 0), '');
            setStat('outputRate', formatSampleRate(stats.output_rate || 0), '');

            // Daemon running status from WebSocket
            const daemonRunning = stats.daemon_running;
            setStatus('daemonStatus', daemonRunning ? 'Running' : 'Stopped', daemonRunning);
            document.getElementById('startBtn').disabled = daemonRunning;
            document.getElementById('stopBtn').disabled = !daemonRunning;
        }

        async function fetchStatus() {
            try {
                const [statusRes, daemonRes] = await Promise.all([
                    fetch(API + '/status'),
                    fetch(API + '/daemon/status')
                ]);
                const status = await statusRes.json();
                const daemon = await daemonRes.json();

                // Daemon status
                setStatus('daemonStatus', daemon.running ? 'Running' : 'Stopped', daemon.running);
                document.getElementById('pidStatus').querySelector('.value').textContent = daemon.pid || '-';
                document.getElementById('pidStatus').classList.remove('ok', 'error');
                document.getElementById('pidStatus').classList.add(daemon.pid ? 'ok' : 'error');
                setStatus('pwStatus', daemon.pipewire_connected ? 'OK' : 'N/A', daemon.pipewire_connected);

                // Stats - clip_rate is now a ratio (0-1), multiply by 100 for percentage
                const clipPct = (status.clip_rate * 100).toFixed(4);
                setStat('clipRate', clipPct + '%', status.clip_rate < 0.001 ? '' : (status.clip_rate < 0.01 ? 'warning' : 'error'));
                setStat('clipCount', formatNumber(status.clip_count), '');
                setStat('totalSamples', formatNumber(status.total_samples), '');
                setStat('eqStatus', status.eq_active ? 'ON' : 'OFF', status.eq_active ? '' : 'error');

                // Sample rates
                setStat('inputRate', formatSampleRate(status.input_rate || 0), '');
                setStat('outputRate', formatSampleRate(status.output_rate || 0), '');

                // System info
                document.getElementById('pidFile').textContent = daemon.pid_file || '-';
                document.getElementById('binaryPath').textContent = daemon.binary_path ? daemon.binary_path.split('/').pop() : '-';
                document.getElementById('alsaDevice').textContent = status.settings.alsa_device || '-';
                document.getElementById('upsampleRatio').textContent = status.settings.upsample_ratio + 'x';

                // Enable/disable buttons based on state
                document.getElementById('startBtn').disabled = daemon.running;
                document.getElementById('stopBtn').disabled = !daemon.running;
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

        function setStat(id, value, level) {
            const el = document.getElementById(id);
            el.querySelector('.value').textContent = value;
            el.classList.remove('warning', 'error');
            if (level) el.classList.add(level);
        }

        function formatNumber(n) {
            if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
            if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
            if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
            return String(n);
        }

        function formatSampleRate(hz) {
            if (hz <= 0) return '-';
            if (hz >= 1000) return (hz / 1000).toFixed(1) + ' kHz';
            return hz + ' Hz';
        }

        function showMessage(text, success) {
            const el = document.getElementById('controlMessage');
            el.textContent = text;
            el.classList.remove('success', 'error');
            el.classList.add(success ? 'success' : 'error');
            setTimeout(() => el.classList.remove('success', 'error'), 4000);
        }

        // EQ Profile display
        let currentEqData = null;

        async function fetchEqProfile() {
            try {
                const res = await fetch(API + '/eq/active');
                const data = await res.json();
                currentEqData = data;
                renderEqProfile(data);
            } catch (e) {
                console.error('Failed to fetch EQ profile:', e);
            }
        }

        function renderEqProfile(data) {
            const nameEl = document.getElementById('eqProfileName');
            const contentEl = document.getElementById('eqContent');
            const copyBtn = document.getElementById('copyEqBtn');

            if (!data.active) {
                nameEl.textContent = '-';
                nameEl.classList.add('eq-inactive');
                contentEl.innerHTML = '<div class="eq-inactive">EQ無効</div>';
                copyBtn.style.display = 'none';
                return;
            }

            nameEl.textContent = data.name || 'Unknown';
            nameEl.classList.remove('eq-inactive');

            // Handle error state (file not found, parse error, etc.)
            if (data.error) {
                contentEl.innerHTML = '<div class="eq-inactive" style="color:#ff6b6b;">エラー: ' + escapeHtml(data.error) + '</div>';
                copyBtn.style.display = 'none';
                return;
            }

            copyBtn.style.display = 'block';

            let html = '';

            if (data.source_type === 'opra') {
                // OPRA section
                html += '<div class="eq-section-label">OPRA (CC BY-SA 4.0)</div>';
                if (data.opra_info) {
                    if (data.opra_info.author) {
                        html += `<div style="font-size:11px;color:#888;margin-bottom:8px;">Author: ${escapeHtml(data.opra_info.author)}</div>`;
                    }
                }
                html += '<div class="eq-filters">' + renderFilters(data.opra_filters) + '</div>';

                // Attribution
                html += '<div class="eq-attribution">Source: <a href="https://github.com/opra-project/OPRA" target="_blank">OPRA Project</a></div>';

                // Original additions section
                if (data.original_filters && data.original_filters.length > 0) {
                    html += '<div class="eq-section-label">オリジナル追加</div>';
                    html += '<div class="eq-filters">' + renderFilters(data.original_filters) + '</div>';
                }
            } else {
                // Custom profile
                html += '<div class="eq-section-label">カスタムプロファイル</div>';
                html += '<div class="eq-filters">' + renderFilters(data.opra_filters) + '</div>';
            }

            contentEl.innerHTML = html;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function renderFilters(filters) {
            // Escape each filter line and join with <br> for proper display
            return filters.map(f => escapeHtml(f)).join('<br>');
        }

        function getEqTextForCopy() {
            if (!currentEqData || !currentEqData.active) return '';
            let lines = [];
            if (currentEqData.source_type === 'opra' && currentEqData.opra_info) {
                lines.push('# OPRA: ' + (currentEqData.opra_info.product || currentEqData.name));
                if (currentEqData.opra_info.author) lines.push('# Author: ' + currentEqData.opra_info.author);
                lines.push('# License: CC BY-SA 4.0');
                lines.push('# Source: https://github.com/opra-project/OPRA');
                lines.push('');
            }
            lines = lines.concat(currentEqData.opra_filters);
            if (currentEqData.original_filters && currentEqData.original_filters.length > 0) {
                lines.push('');
                lines.push('# オリジナル追加');
                lines = lines.concat(currentEqData.original_filters);
            }
            return lines.join(String.fromCharCode(10));
        }

        document.getElementById('copyEqBtn').addEventListener('click', async () => {
            const text = getEqTextForCopy();
            if (!text) return;
            try {
                await navigator.clipboard.writeText(text);
                const btn = document.getElementById('copyEqBtn');
                btn.textContent = 'Copied!';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.textContent = 'Copy';
                    btn.classList.remove('copied');
                }, 2000);
            } catch (e) {
                console.error('Failed to copy:', e);
            }
        });

        document.getElementById('startBtn').addEventListener('click', async () => {
            const btn = document.getElementById('startBtn');
            btn.disabled = true;
            try {
                const res = await fetch(API + '/daemon/start', { method: 'POST' });
                const data = await res.json();
                showMessage(data.message, data.success);
                setTimeout(fetchStatus, 1000);
            } catch (e) {
                showMessage('Error: ' + e.message, false);
            }
            btn.disabled = false;
        });

        document.getElementById('stopBtn').addEventListener('click', async () => {
            const btn = document.getElementById('stopBtn');
            btn.disabled = true;
            try {
                const res = await fetch(API + '/daemon/stop', { method: 'POST' });
                const data = await res.json();
                showMessage(data.message, data.success);
                setTimeout(fetchStatus, 1000);
            } catch (e) {
                showMessage('Error: ' + e.message, false);
            }
            btn.disabled = false;
        });

        document.getElementById('restartBtn').addEventListener('click', async () => {
            const btn = document.getElementById('restartBtn');
            btn.disabled = true;
            btn.textContent = 'Restarting...';
            try {
                const res = await fetch(API + '/daemon/restart', { method: 'POST' });
                const data = await res.json();
                showMessage(data.message, data.success);
                setTimeout(fetchStatus, 2000);
            } catch (e) {
                showMessage('Error: ' + e.message, false);
            }
            btn.disabled = false;
            btn.textContent = 'Restart';
        });

        // ============================================================
        // Custom EQ Upload
        // ============================================================
        let pendingFile = null;
        let validationResult = null;

        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('eqFileInput');
        const uploadInfo = document.getElementById('uploadInfo');
        const uploadActions = document.getElementById('uploadActions');
        const uploadBtn = document.getElementById('uploadBtn');
        const cancelUploadBtn = document.getElementById('cancelUploadBtn');

        uploadZone.addEventListener('click', () => fileInput.click());

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) handleFileSelect(files[0]);
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
        });

        async function handleFileSelect(file) {
            pendingFile = file;
            validationResult = null;

            // Show loading state
            uploadInfo.style.display = 'block';
            uploadInfo.className = 'upload-info';
            uploadInfo.innerHTML = '<div class="filename">' + escapeHtml(file.name) + '</div><div class="details">検証中...</div>';
            uploadActions.style.display = 'none';

            // Validate via API
            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch(API + '/eq/validate', { method: 'POST', body: formData });
                if (res.ok) {
                    validationResult = await res.json();
                    renderValidationResult(validationResult, file.name);
                } else {
                    const err = await res.json();
                    showUploadError(err.detail || 'Validation failed');
                }
            } catch (e) {
                showUploadError('検証エラー: ' + e.message);
            }
        }

        function renderValidationResult(result, filename) {
            let html = '<div class="filename">' + escapeHtml(result.filename || filename) + '</div>';
            html += '<div class="details">';
            html += 'フィルター数: ' + result.filter_count;
            if (result.preamp_db !== null) html += ' / Preamp: ' + result.preamp_db + 'dB';
            html += ' / サイズ: ' + formatBytes(result.size_bytes);
            html += '</div>';

            if (result.errors && result.errors.length > 0) {
                html += '<div class="errors">エラー: ' + result.errors.map(escapeHtml).join(', ') + '</div>';
            }
            if (result.warnings && result.warnings.length > 0) {
                html += '<div class="warnings">警告: ' + result.warnings.map(escapeHtml).join(', ') + '</div>';
            }
            if (result.file_exists) {
                html += '<div class="warnings">⚠️ 同名のファイルが既に存在します（上書きされます）</div>';
            }

            uploadInfo.innerHTML = html;
            uploadInfo.className = 'upload-info' + (result.errors?.length ? ' error' : (result.warnings?.length ? ' warning' : ''));

            uploadActions.style.display = 'flex';
            uploadBtn.disabled = !result.valid;
            uploadBtn.textContent = result.file_exists ? '上書きアップロード' : 'アップロード';
        }

        function showUploadError(msg) {
            uploadInfo.className = 'upload-info error';
            uploadInfo.innerHTML = '<div class="errors">' + escapeHtml(msg) + '</div>';
            uploadActions.style.display = 'flex';
            uploadBtn.disabled = true;
        }

        function showUploadMessage(text, success) {
            const el = document.getElementById('uploadMessage');
            el.textContent = text;
            el.classList.remove('success', 'error');
            el.classList.add(success ? 'success' : 'error');
            setTimeout(() => el.classList.remove('success', 'error'), 4000);
        }

        function formatBytes(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
        }

        cancelUploadBtn.addEventListener('click', () => {
            pendingFile = null;
            validationResult = null;
            uploadInfo.style.display = 'none';
            uploadActions.style.display = 'none';
            fileInput.value = '';
        });

        uploadBtn.addEventListener('click', async () => {
            if (!pendingFile || !validationResult || !validationResult.valid) return;

            uploadBtn.disabled = true;
            uploadBtn.textContent = 'アップロード中...';

            const formData = new FormData();
            formData.append('file', pendingFile);

            const overwrite = validationResult.file_exists ? '?overwrite=true' : '';

            try {
                const res = await fetch(API + '/eq/import' + overwrite, { method: 'POST', body: formData });
                const data = await res.json();
                if (res.ok && data.success) {
                    showUploadMessage(data.message, true);
                    // Reset upload UI
                    pendingFile = null;
                    validationResult = null;
                    uploadInfo.style.display = 'none';
                    uploadActions.style.display = 'none';
                    fileInput.value = '';
                    // Refresh profile list
                    fetchProfiles();
                } else {
                    showUploadMessage(data.detail || data.message || 'Upload failed', false);
                }
            } catch (e) {
                showUploadMessage('アップロードエラー: ' + e.message, false);
            }
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'アップロード';
        });

        // ============================================================
        // Profile List
        // ============================================================
        async function fetchProfiles() {
            try {
                const res = await fetch(API + '/eq/profiles');
                const data = await res.json();
                renderProfiles(data.profiles);
            } catch (e) {
                console.error('Failed to fetch profiles:', e);
                document.getElementById('profileList').innerHTML = '<div class="empty-state">読み込みエラー</div>';
            }
        }

        function renderProfiles(profiles) {
            const container = document.getElementById('profileList');
            if (!profiles || profiles.length === 0) {
                container.innerHTML = '<div class="empty-state">保存済みプロファイルがありません</div>';
                return;
            }

            container.innerHTML = profiles.map(p => {
                const typeBadge = p.type === 'opra'
                    ? '<span class="type-badge opra">OPRA</span>'
                    : '<span class="type-badge custom">Custom</span>';
                const meta = p.filter_count + ' filters / ' + formatBytes(p.size);
                return `
                    <div class="profile-item" data-name="${escapeHtml(p.name)}">
                        <div class="info">
                            <div class="name">${escapeHtml(p.name)}${typeBadge}</div>
                            <div class="meta">${meta}</div>
                        </div>
                        <div class="actions">
                            <button class="btn-success btn-small apply-btn">適用</button>
                            <button class="btn-danger btn-small delete-btn">削除</button>
                        </div>
                    </div>
                `;
            }).join('');

            // Add event listeners
            container.querySelectorAll('.apply-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    const name = e.target.closest('.profile-item').dataset.name;
                    await applyProfile(name, e.target);
                });
            });

            container.querySelectorAll('.delete-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    const name = e.target.closest('.profile-item').dataset.name;
                    if (confirm('プロファイル "' + name + '" を削除しますか？')) {
                        await deleteProfile(name, e.target);
                    }
                });
            });
        }

        async function applyProfile(name, btn) {
            btn.disabled = true;
            btn.textContent = '適用中...';
            try {
                const res = await fetch(API + '/eq/activate/' + encodeURIComponent(name), { method: 'POST' });
                const data = await res.json();
                showUploadMessage(data.message, data.success);
                if (data.success && data.restart_required) {
                    btn.textContent = '再起動中...';
                    await fetch(API + '/restart', { method: 'POST' });
                    setTimeout(() => {
                        fetchStatus();
                        fetchEqProfile();
                    }, 2000);
                }
            } catch (e) {
                showUploadMessage('適用エラー: ' + e.message, false);
            }
            btn.disabled = false;
            btn.textContent = '適用';
        }

        async function deleteProfile(name, btn) {
            btn.disabled = true;
            btn.textContent = '削除中...';
            try {
                const res = await fetch(API + '/eq/profiles/' + encodeURIComponent(name), { method: 'DELETE' });
                const data = await res.json();
                showUploadMessage(data.message, data.success);
                if (data.success) {
                    fetchProfiles();
                    fetchEqProfile();
                }
            } catch (e) {
                showUploadMessage('削除エラー: ' + e.message, false);
            }
            btn.disabled = false;
            btn.textContent = '削除';
        }

        // Initial load and auto-refresh
        fetchStatus();
        fetchEqProfile();
        fetchProfiles();
        // Full status refresh every 5 seconds (for daemon info, settings, etc.)
        setInterval(fetchStatus, 5000);
        // EQ profile refresh every 10 seconds (less frequent, doesn't change often)
        setInterval(fetchEqProfile, 10000);
        // Profile list refresh every 30 seconds
        setInterval(fetchProfiles, 30000);
        // Connect WebSocket for real-time stats (1 second updates)
        connectWebSocket();
    </script>
</body>
</html>
"""
