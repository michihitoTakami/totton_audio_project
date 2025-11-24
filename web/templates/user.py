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
                <option value="minimum">Minimum Phase (推奨)</option>
                <option value="linear">Linear Phase</option>
            </select>
        </div>
        <div id="phaseWarning" class="warning-banner">
            ⚠️ Linear phaseはレイテンシが約1秒あります。リアルタイム用途にはMinimum phaseを推奨します。
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

    <script>
        const API = '';
        let currentAlsaDevice = '';
        let deviceList = [];

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
                setStatus('pwStatus', data.pipewire_connected ? 'OK' : 'N/A', data.pipewire_connected);
                setStatus('eqStatus', data.eq_active ? 'ON' : 'OFF', data.eq_active);

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

        function showMessage(text, success) {
            const el = document.getElementById('settingsMessage');
            el.textContent = text;
            el.classList.remove('success', 'error');
            el.classList.add(success ? 'success' : 'error');
            setTimeout(() => el.classList.remove('success', 'error'), 4000);
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
                const data = await res.json();
                renderOpraResults(data.results);
            } catch (e) {
                console.error('OPRA search failed:', e);
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
            } catch (e) {
                console.error('Failed to fetch phase type:', e);
                select.disabled = true;
                warning.classList.remove('visible');
            }
        }

        function updatePhaseWarning(phaseType, apiWarning) {
            const warning = document.getElementById('phaseWarning');
            if (phaseType === 'linear') {
                // Use API warning if provided, otherwise use default
                if (apiWarning) {
                    warning.textContent = '⚠️ ' + apiWarning;
                }
                warning.classList.add('visible');
            } else {
                warning.classList.remove('visible');
            }
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
                    showPhaseMessage('Phase type updated to ' + newPhaseType, true);
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

        // Initial load
        fetchDevices();
        fetchStatus();
        fetchPhaseType();
        setInterval(fetchStatus, 5000);
        setInterval(fetchPhaseType, 5000);
    </script>
</body>
</html>
"""
