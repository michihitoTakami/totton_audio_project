"""Magic Box UI Mockup Templates."""


def get_headphones_html() -> str:
    """Return headphone selection page HTML"""
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Magic Box - ヘッドホン選択</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            display: flex;
            min-height: 100vh;
        }

        /* Sidebar (same as dashboard) */
        .sidebar {
            width: 240px;
            background: #0f1419;
            border-right: 1px solid #1e2a3a;
            display: flex;
            flex-direction: column;
            padding: 20px 0;
        }
        .logo {
            padding: 0 20px 30px;
            border-bottom: 1px solid #1e2a3a;
            margin-bottom: 20px;
        }
        .logo h1 {
            color: #00d4ff;
            font-size: 1.4em;
            margin-bottom: 4px;
        }
        .logo .tagline {
            color: #666;
            font-size: 11px;
        }
        .nav-item {
            padding: 12px 20px;
            color: #aaa;
            text-decoration: none;
            display: block;
            transition: all 0.2s;
            border-left: 3px solid transparent;
        }
        .nav-item:hover {
            background: #1a2332;
            color: #00d4ff;
        }
        .nav-item.active {
            background: #1a2332;
            color: #00d4ff;
            border-left-color: #00d4ff;
        }

        /* Main content */
        .main-content {
            flex: 1;
            padding: 40px;
            overflow-y: auto;
        }
        .page-header {
            margin-bottom: 30px;
        }
        .page-header h2 {
            font-size: 2em;
            margin-bottom: 8px;
        }
        .page-header .subtitle {
            color: #888;
            font-size: 14px;
        }

        /* Search box */
        .search-box {
            background: rgba(22, 33, 62, 0.6);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(0, 212, 255, 0.1);
        }
        .search-input {
            width: 100%;
            padding: 14px;
            border: 1px solid #1e2a3a;
            border-radius: 8px;
            background: #0f3460;
            color: #eee;
            font-size: 14px;
        }
        .search-input:focus {
            outline: none;
            border-color: #00d4ff;
        }

        /* Headphone grid */
        .headphone-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        .headphone-card {
            background: rgba(22, 33, 62, 0.6);
            border-radius: 12px;
            padding: 20px;
            border: 2px solid transparent;
            cursor: pointer;
            transition: all 0.2s;
        }
        .headphone-card:hover {
            border-color: #00d4ff;
            transform: translateY(-4px);
        }
        .headphone-card.selected {
            border-color: #00ff88;
            background: rgba(0, 255, 136, 0.05);
        }
        .headphone-card .brand {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .headphone-card .model {
            font-size: 18px;
            font-weight: 600;
            color: #00d4ff;
            margin-bottom: 12px;
        }
        .headphone-card .info {
            font-size: 12px;
            color: #aaa;
            margin-bottom: 4px;
        }
        .headphone-card .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 10px;
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
            margin-top: 8px;
        }

        /* Apply button */
        .apply-section {
            position: fixed;
            bottom: 0;
            left: 240px;
            right: 0;
            background: rgba(15, 20, 25, 0.95);
            border-top: 1px solid #1e2a3a;
            padding: 20px 40px;
            backdrop-filter: blur(10px);
        }
        .apply-btn {
            width: 100%;
            padding: 16px;
            border: none;
            border-radius: 8px;
            background: #00ff88;
            color: #000;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .apply-btn:hover {
            background: #00cc6a;
        }
        .apply-btn:disabled {
            background: #333;
            color: #666;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <aside class="sidebar">
        <div class="logo">
            <h1>Magic Box</h1>
            <div class="tagline">Ultimate Audio Experience</div>
        </div>
        <nav>
            <a href="/mockup/dashboard" class="nav-item">📊 ダッシュボード</a>
            <a href="/mockup/headphones" class="nav-item active">🎧 ヘッドホン選択</a>
            <a href="/mockup/eq" class="nav-item">🎚️ EQ設定</a>
            <a href="/mockup/system" class="nav-item">⚙️ システム設定</a>
            <a href="/mockup/status" class="nav-item">📈 ステータス監視</a>
            <a href="/mockup/rtp" class="nav-item">🌐 RTP設定</a>
        </nav>
    </aside>

    <main class="main-content">
        <header class="page-header">
            <h2>ヘッドホン選択</h2>
            <p class="subtitle">OPRAプロジェクトのデータベースから最適な補正プロファイルを選択</p>
        </header>

        <div class="search-box">
            <input type="text" class="search-input" placeholder="🔍 ヘッドホンを検索 (例: Sony, Sennheiser, Audio-Technica...)">
        </div>

        <div class="headphone-grid">
            <div class="headphone-card selected">
                <div class="brand">Sony</div>
                <div class="model">MDR-Z1R</div>
                <div class="info">Target: KB5000_7 (推奨)</div>
                <div class="info">Source: OPRA AutoEQ InnerFidelity</div>
                <span class="badge">現在選択中</span>
            </div>

            <div class="headphone-card">
                <div class="brand">Sennheiser</div>
                <div class="model">HD 800 S</div>
                <div class="info">Target: KB5000_7</div>
                <div class="info">Source: OPRA AutoEQ</div>
            </div>

            <div class="headphone-card">
                <div class="brand">Audio-Technica</div>
                <div class="model">ATH-M50x</div>
                <div class="info">Target: KB5000_7</div>
                <div class="info">Source: OPRA AutoEQ</div>
            </div>

            <div class="headphone-card">
                <div class="brand">Focal</div>
                <div class="model">Clear</div>
                <div class="info">Target: KB5000_7</div>
                <div class="info">Source: OPRA AutoEQ</div>
            </div>

            <div class="headphone-card">
                <div class="brand">Beyerdynamic</div>
                <div class="model">DT 1990 Pro</div>
                <div class="info">Target: KB5000_7</div>
                <div class="info">Source: OPRA AutoEQ</div>
            </div>

            <div class="headphone-card">
                <div class="brand">Audeze</div>
                <div class="model">LCD-X</div>
                <div class="info">Target: KB5000_7</div>
                <div class="info">Source: OPRA AutoEQ</div>
            </div>
        </div>

        <div style="height: 100px;"></div>
    </main>

    <div class="apply-section">
        <button class="apply-btn">このヘッドホンで適用</button>
    </div>
</body>
</html>
"""


def get_eq_html() -> str:
    """Return EQ settings page HTML"""
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Magic Box - EQ設定</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            display: flex;
            min-height: 100vh;
        }

        /* Sidebar (same as dashboard) */
        .sidebar {
            width: 240px;
            background: #0f1419;
            border-right: 1px solid #1e2a3a;
            display: flex;
            flex-direction: column;
            padding: 20px 0;
        }
        .logo {
            padding: 0 20px 30px;
            border-bottom: 1px solid #1e2a3a;
            margin-bottom: 20px;
        }
        .logo h1 {
            color: #00d4ff;
            font-size: 1.4em;
            margin-bottom: 4px;
        }
        .logo .tagline {
            color: #666;
            font-size: 11px;
        }
        .nav-item {
            padding: 12px 20px;
            color: #aaa;
            text-decoration: none;
            display: block;
            transition: all 0.2s;
            border-left: 3px solid transparent;
        }
        .nav-item:hover {
            background: #1a2332;
            color: #00d4ff;
        }
        .nav-item.active {
            background: #1a2332;
            color: #00d4ff;
            border-left-color: #00d4ff;
        }

        /* Main content */
        .main-content {
            flex: 1;
            padding: 40px;
            overflow-y: auto;
        }
        .page-header {
            margin-bottom: 30px;
        }
        .page-header h2 {
            font-size: 2em;
            margin-bottom: 8px;
        }
        .page-header .subtitle {
            color: #888;
            font-size: 14px;
        }

        /* Current profile */
        .current-profile {
            background: rgba(22, 33, 62, 0.6);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(0, 212, 255, 0.1);
        }
        .current-profile h3 {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        .current-profile .profile-name {
            font-size: 20px;
            color: #00d4ff;
            margin-bottom: 8px;
        }
        .current-profile .profile-info {
            font-size: 12px;
            color: #aaa;
        }

        /* EQ controls */
        .eq-section {
            background: rgba(22, 33, 62, 0.6);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(0, 212, 255, 0.1);
        }
        .eq-section h3 {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 20px;
        }

        /* Preamp control */
        .preamp-control {
            background: #0f3460;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .preamp-control label {
            display: block;
            font-size: 13px;
            color: #aaa;
            margin-bottom: 12px;
        }
        .preamp-control .slider-container {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .preamp-control input[type="range"] {
            flex: 1;
            height: 6px;
            border-radius: 3px;
            background: #1a2332;
            outline: none;
        }
        .preamp-control .value {
            font-size: 18px;
            font-weight: 600;
            color: #00d4ff;
            min-width: 70px;
            text-align: right;
        }

        /* Band controls */
        .band-list {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .band-item {
            background: #0f3460;
            padding: 16px;
            border-radius: 8px;
            display: grid;
            grid-template-columns: 80px 1fr 100px;
            gap: 16px;
            align-items: center;
        }
        .band-item .freq {
            font-size: 14px;
            font-weight: 600;
            color: #00d4ff;
        }
        .band-item .type {
            font-size: 11px;
            color: #666;
        }
        .band-item .slider-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .band-item input[type="range"] {
            flex: 1;
            height: 4px;
            border-radius: 2px;
            background: #1a2332;
        }
        .band-item .gain-value {
            font-size: 14px;
            font-weight: 600;
            color: #00ff88;
            min-width: 60px;
            text-align: right;
        }
        .band-item .toggle {
            width: 44px;
            height: 24px;
            background: #1a2332;
            border-radius: 12px;
            position: relative;
            cursor: pointer;
        }
        .band-item .toggle::after {
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #666;
            top: 2px;
            left: 2px;
            transition: all 0.2s;
        }
        .band-item .toggle.active {
            background: #00ff88;
        }
        .band-item .toggle.active::after {
            left: 22px;
            background: #fff;
        }

        /* Action buttons */
        .action-buttons {
            display: flex;
            gap: 12px;
            margin-top: 24px;
        }
        .action-buttons button {
            flex: 1;
            padding: 14px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary {
            background: #00ff88;
            color: #000;
        }
        .btn-primary:hover {
            background: #00cc6a;
        }
        .btn-secondary {
            background: #0f3460;
            color: #eee;
        }
        .btn-secondary:hover {
            background: #1a4b7c;
        }
    </style>
</head>
<body>
    <aside class="sidebar">
        <div class="logo">
            <h1>Magic Box</h1>
            <div class="tagline">Ultimate Audio Experience</div>
        </div>
        <nav>
            <a href="/mockup/dashboard" class="nav-item">📊 ダッシュボード</a>
            <a href="/mockup/headphones" class="nav-item">🎧 ヘッドホン選択</a>
            <a href="/mockup/eq" class="nav-item active">🎚️ EQ設定</a>
            <a href="/mockup/system" class="nav-item">⚙️ システム設定</a>
            <a href="/mockup/status" class="nav-item">📈 ステータス監視</a>
            <a href="/mockup/rtp" class="nav-item">🌐 RTP設定</a>
        </nav>
    </aside>

    <main class="main-content">
        <header class="page-header">
            <h2>EQ設定</h2>
            <p class="subtitle">ヘッドホン補正の微調整とプリアンプ設定</p>
        </header>

        <div class="current-profile">
            <h3>現在のプロファイル</h3>
            <div class="profile-name">Sony MDR-Z1R (KB5000_7)</div>
            <div class="profile-info">OPRA AutoEQ InnerFidelity + Custom Filter 11</div>
        </div>

        <div class="eq-section">
            <h3>プリアンプ (Headroom)</h3>
            <div class="preamp-control">
                <label>全体のゲイン調整 (クリッピング防止)</label>
                <div class="slider-container">
                    <input type="range" min="-20" max="0" value="-6.8" step="0.1">
                    <span class="value">-6.8 dB</span>
                </div>
            </div>
        </div>

        <div class="eq-section">
            <h3>パラメトリックEQ (11 Bands)</h3>
            <div class="band-list">
                <div class="band-item">
                    <div>
                        <div class="freq">22 Hz</div>
                        <div class="type">LOW SHELF</div>
                    </div>
                    <div class="slider-container">
                        <input type="range" min="-12" max="12" value="6.3" step="0.1">
                        <span class="gain-value">+6.3 dB</span>
                    </div>
                    <div class="toggle active"></div>
                </div>

                <div class="band-item">
                    <div>
                        <div class="freq">150 Hz</div>
                        <div class="type">PEAK Q 0.7</div>
                    </div>
                    <div class="slider-container">
                        <input type="range" min="-12" max="12" value="3.4" step="0.1">
                        <span class="gain-value">+3.4 dB</span>
                    </div>
                    <div class="toggle active"></div>
                </div>

                <div class="band-item">
                    <div>
                        <div class="freq">683 Hz</div>
                        <div class="type">PEAK Q 1.8</div>
                    </div>
                    <div class="slider-container">
                        <input type="range" min="-12" max="12" value="-1.3" step="0.1">
                        <span class="gain-value">-1.3 dB</span>
                    </div>
                    <div class="toggle active"></div>
                </div>

                <div class="band-item">
                    <div>
                        <div class="freq">2184 Hz</div>
                        <div class="type">PEAK Q 2.9</div>
                    </div>
                    <div class="slider-container">
                        <input type="range" min="-12" max="12" value="-3.1" step="0.1">
                        <span class="gain-value">-3.1 dB</span>
                    </div>
                    <div class="toggle active"></div>
                </div>

                <div class="band-item">
                    <div>
                        <div class="freq">3043 Hz</div>
                        <div class="type">PEAK Q 3.0</div>
                    </div>
                    <div class="slider-container">
                        <input type="range" min="-12" max="12" value="1.8" step="0.1">
                        <span class="gain-value">+1.8 dB</span>
                    </div>
                    <div class="toggle active"></div>
                </div>

                <div class="band-item">
                    <div>
                        <div class="freq">5366 Hz</div>
                        <div class="type">PEAK Q 1.5</div>
                    </div>
                    <div class="slider-container">
                        <input type="range" min="-12" max="12" value="2.8" step="0.1">
                        <span class="gain-value">+2.8 dB</span>
                    </div>
                    <div class="toggle active"></div>
                </div>
            </div>

            <div class="action-buttons">
                <button class="btn-secondary">リセット</button>
                <button class="btn-secondary">デフォルトに戻す</button>
                <button class="btn-primary">適用</button>
            </div>
        </div>
    </main>
</body>
</html>
"""


def get_system_html() -> str:
    """Return system settings page HTML"""
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Magic Box - システム設定</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            display: flex;
            min-height: 100vh;
        }

        /* Sidebar */
        .sidebar {
            width: 240px;
            background: #0f1419;
            border-right: 1px solid #1e2a3a;
            display: flex;
            flex-direction: column;
            padding: 20px 0;
        }
        .logo {
            padding: 0 20px 30px;
            border-bottom: 1px solid #1e2a3a;
            margin-bottom: 20px;
        }
        .logo h1 {
            color: #00d4ff;
            font-size: 1.4em;
            margin-bottom: 4px;
        }
        .logo .tagline {
            color: #666;
            font-size: 11px;
        }
        .nav-item {
            padding: 12px 20px;
            color: #aaa;
            text-decoration: none;
            display: block;
            transition: all 0.2s;
            border-left: 3px solid transparent;
        }
        .nav-item:hover {
            background: #1a2332;
            color: #00d4ff;
        }
        .nav-item.active {
            background: #1a2332;
            color: #00d4ff;
            border-left-color: #00d4ff;
        }

        /* Main content */
        .main-content {
            flex: 1;
            padding: 40px;
            overflow-y: auto;
        }
        .page-header {
            margin-bottom: 30px;
        }
        .page-header h2 {
            font-size: 2em;
            margin-bottom: 8px;
        }
        .page-header .subtitle {
            color: #888;
            font-size: 14px;
        }

        /* Settings section */
        .settings-section {
            background: rgba(22, 33, 62, 0.6);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(0, 212, 255, 0.1);
        }
        .settings-section h3 {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 20px;
            letter-spacing: 1px;
        }

        /* Setting item */
        .setting-item {
            background: #0f3460;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 12px;
        }
        .setting-item:last-child {
            margin-bottom: 0;
        }
        .setting-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .setting-header .title {
            font-size: 14px;
            font-weight: 600;
            color: #eee;
        }
        .setting-header .value {
            font-size: 14px;
            color: #00d4ff;
            font-weight: 600;
        }
        .setting-item .description {
            font-size: 12px;
            color: #888;
            margin-bottom: 12px;
        }

        /* Dropdown */
        select {
            width: 100%;
            padding: 12px;
            border: 1px solid #1e2a3a;
            border-radius: 6px;
            background: #1a2332;
            color: #eee;
            font-size: 14px;
            cursor: pointer;
        }
        select:focus {
            outline: none;
            border-color: #00d4ff;
        }

        /* Toggle switch */
        .toggle-switch {
            width: 50px;
            height: 26px;
            background: #1a2332;
            border-radius: 13px;
            position: relative;
            cursor: pointer;
            transition: background 0.2s;
        }
        .toggle-switch::after {
            content: '';
            position: absolute;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: #666;
            top: 2px;
            left: 2px;
            transition: all 0.2s;
        }
        .toggle-switch.active {
            background: #00ff88;
        }
        .toggle-switch.active::after {
            left: 26px;
            background: #fff;
        }

        /* Action buttons */
        .action-buttons {
            display: flex;
            gap: 12px;
        }
        .action-buttons button {
            flex: 1;
            padding: 14px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary {
            background: #00ff88;
            color: #000;
        }
        .btn-primary:hover {
            background: #00cc6a;
        }
        .btn-danger {
            background: #ff4444;
            color: #fff;
        }
        .btn-danger:hover {
            background: #cc3333;
        }
    </style>
</head>
<body>
    <aside class="sidebar">
        <div class="logo">
            <h1>Magic Box</h1>
            <div class="tagline">Ultimate Audio Experience</div>
        </div>
        <nav>
            <a href="/mockup/dashboard" class="nav-item">📊 ダッシュボード</a>
            <a href="/mockup/headphones" class="nav-item">🎧 ヘッドホン選択</a>
            <a href="/mockup/eq" class="nav-item">🎚️ EQ設定</a>
            <a href="/mockup/system" class="nav-item active">⚙️ システム設定</a>
            <a href="/mockup/status" class="nav-item">📈 ステータス監視</a>
            <a href="/mockup/rtp" class="nav-item">🌐 RTP設定</a>
        </nav>
    </aside>

    <main class="main-content">
        <header class="page-header">
            <h2>システム設定</h2>
            <p class="subtitle">オーディオエンジンの詳細設定</p>
        </header>

        <div class="settings-section">
            <h3>フィルタ設定</h3>

            <div class="setting-item">
                <div class="setting-header">
                    <span class="title">位相タイプ</span>
                    <span class="value">Minimum Phase</span>
                </div>
                <div class="description">Minimum Phaseは音楽再生に最適（プリリンギングなし）。Linear Phaseはマスタリング向け。</div>
                <select>
                    <option selected>Minimum Phase (推奨)</option>
                    <option>Linear Phase</option>
                </select>
            </div>

            <div class="setting-item">
                <div class="setting-header">
                    <span class="title">フィルタタップ数</span>
                    <span class="value">640,000 taps</span>
                </div>
                <div class="description">より多いタップ数 = より高い品質、ただしGPU負荷も増加</div>
                <select>
                    <option>160,000 taps (軽量)</option>
                    <option>320,000 taps (標準)</option>
                    <option selected>640,000 taps (高品質)</option>
                    <option>2,000,000 taps (実験的)</option>
                </select>
            </div>
        </div>

        <div class="settings-section">
            <h3>アップサンプリング</h3>

            <div class="setting-item">
                <div class="setting-header">
                    <span class="title">自動ネゴシエーション</span>
                    <div class="toggle-switch active"></div>
                </div>
                <div class="description">DAC能力に応じて最適なアップサンプリング倍率を自動決定</div>
            </div>

            <div class="setting-item">
                <div class="setting-header">
                    <span class="title">最大アップサンプリング倍率</span>
                    <span class="value">16x</span>
                </div>
                <div class="description">44.1kHz → 705.6kHz / 48kHz → 768kHz</div>
                <select>
                    <option>2x</option>
                    <option>4x</option>
                    <option>8x</option>
                    <option selected>16x</option>
                </select>
            </div>
        </div>

        <div class="settings-section">
            <h3>安全機能</h3>

            <div class="setting-item">
                <div class="setting-header">
                    <span class="title">ソフトミュート</span>
                    <div class="toggle-switch active"></div>
                </div>
                <div class="description">レート切り替え時のポップノイズ防止（クロスフェード）</div>
            </div>

            <div class="setting-item">
                <div class="setting-header">
                    <span class="title">動的フォールバック</span>
                    <div class="toggle-switch active"></div>
                </div>
                <div class="description">GPU負荷過多時、自動的に軽量モードへ移行</div>
            </div>
        </div>

        <div class="settings-section">
            <h3>システム管理</h3>
            <div class="action-buttons">
                <button class="btn-danger">エンジン再起動</button>
                <button class="btn-primary">設定を保存</button>
            </div>
        </div>
    </main>
</body>
</html>
"""


def get_status_html() -> str:
    """Return status monitoring page HTML"""
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Magic Box - ステータス監視</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            display: flex;
            min-height: 100vh;
        }

        /* Sidebar */
        .sidebar {
            width: 240px;
            background: #0f1419;
            border-right: 1px solid #1e2a3a;
            display: flex;
            flex-direction: column;
            padding: 20px 0;
        }
        .logo {
            padding: 0 20px 30px;
            border-bottom: 1px solid #1e2a3a;
            margin-bottom: 20px;
        }
        .logo h1 {
            color: #00d4ff;
            font-size: 1.4em;
            margin-bottom: 4px;
        }
        .logo .tagline {
            color: #666;
            font-size: 11px;
        }
        .nav-item {
            padding: 12px 20px;
            color: #aaa;
            text-decoration: none;
            display: block;
            transition: all 0.2s;
            border-left: 3px solid transparent;
        }
        .nav-item:hover {
            background: #1a2332;
            color: #00d4ff;
        }
        .nav-item.active {
            background: #1a2332;
            color: #00d4ff;
            border-left-color: #00d4ff;
        }

        /* Main content */
        .main-content {
            flex: 1;
            padding: 40px;
            overflow-y: auto;
        }
        .page-header {
            margin-bottom: 30px;
        }
        .page-header h2 {
            font-size: 2em;
            margin-bottom: 8px;
        }
        .page-header .subtitle {
            color: #888;
            font-size: 14px;
        }

        /* Metrics grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .metric-card {
            background: rgba(22, 33, 62, 0.6);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(0, 212, 255, 0.1);
        }
        .metric-card .label {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .metric-card .value {
            font-size: 32px;
            font-weight: 600;
            color: #00d4ff;
            margin-bottom: 4px;
        }
        .metric-card .unit {
            font-size: 14px;
            color: #666;
        }
        .metric-card.ok .value { color: #00ff88; }
        .metric-card.warning .value { color: #ffaa00; }
        .metric-card.error .value { color: #ff4444; }

        /* Chart placeholder */
        .chart-section {
            background: rgba(22, 33, 62, 0.6);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(0, 212, 255, 0.1);
        }
        .chart-section h3 {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 16px;
            letter-spacing: 1px;
        }
        .chart-placeholder {
            height: 200px;
            background: #0f3460;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-size: 14px;
        }

        /* Audio path */
        .audio-path {
            background: rgba(22, 33, 62, 0.6);
            border-radius: 12px;
            padding: 24px;
            border: 1px solid rgba(0, 212, 255, 0.1);
        }
        .audio-path h3 {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 20px;
            letter-spacing: 1px;
        }
        .path-flow {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }
        .path-node {
            flex: 1;
            background: #0f3460;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        .path-node .node-title {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .path-node .node-value {
            font-size: 16px;
            font-weight: 600;
            color: #00d4ff;
        }
        .path-node.active {
            border: 2px solid #00ff88;
        }
        .path-arrow {
            font-size: 24px;
            color: #666;
        }
    </style>
</head>
<body>
    <aside class="sidebar">
        <div class="logo">
            <h1>Magic Box</h1>
            <div class="tagline">Ultimate Audio Experience</div>
        </div>
        <nav>
            <a href="/mockup/dashboard" class="nav-item">📊 ダッシュボード</a>
            <a href="/mockup/headphones" class="nav-item">🎧 ヘッドホン選択</a>
            <a href="/mockup/eq" class="nav-item">🎚️ EQ設定</a>
            <a href="/mockup/system" class="nav-item">⚙️ システム設定</a>
            <a href="/mockup/status" class="nav-item active">📈 ステータス監視</a>
            <a href="/mockup/rtp" class="nav-item">🌐 RTP設定</a>
        </nav>
    </aside>

    <main class="main-content">
        <header class="page-header">
            <h2>ステータス監視</h2>
            <p class="subtitle">リアルタイムパフォーマンス監視</p>
        </header>

        <div class="metrics-grid">
            <div class="metric-card ok">
                <div class="label">Engine Status</div>
                <div class="value">Running</div>
                <div class="unit">正常動作中</div>
            </div>

            <div class="metric-card">
                <div class="label">GPU Usage</div>
                <div class="value">34%</div>
                <div class="unit">RTX 2070 Super</div>
            </div>

            <div class="metric-card ok">
                <div class="label">Processing Speed</div>
                <div class="value">28.4x</div>
                <div class="unit">Realtime</div>
            </div>

            <div class="metric-card">
                <div class="label">Buffer Latency</div>
                <div class="value">12.8</div>
                <div class="unit">ms</div>
            </div>

            <div class="metric-card">
                <div class="label">VRAM Usage</div>
                <div class="value">2.1</div>
                <div class="unit">GB / 8 GB</div>
            </div>

            <div class="metric-card ok">
                <div class="label">XRUN Count</div>
                <div class="value">0</div>
                <div class="unit">No errors</div>
            </div>
        </div>

        <div class="chart-section">
            <h3>GPU Usage History</h3>
            <div class="chart-placeholder">
                📊 Chart will be rendered here (Canvas/WebGL)
            </div>
        </div>

        <div class="chart-section">
            <h3>Latency History</h3>
            <div class="chart-placeholder">
                📊 Chart will be rendered here (Canvas/WebGL)
            </div>
        </div>

        <div class="audio-path">
            <h3>Audio Signal Path</h3>
            <div class="path-flow">
                <div class="path-node active">
                    <div class="node-title">Input</div>
                    <div class="node-value">44.1 kHz</div>
                </div>
                <div class="path-arrow">→</div>
                <div class="path-node active">
                    <div class="node-title">PipeWire</div>
                    <div class="node-value">Active</div>
                </div>
                <div class="path-arrow">→</div>
                <div class="path-node active">
                    <div class="node-title">GPU Engine</div>
                    <div class="node-value">640k FIR</div>
                </div>
                <div class="path-arrow">→</div>
                <div class="path-node active">
                    <div class="node-title">Resampler</div>
                    <div class="node-value">libsoxr VHQ</div>
                </div>
                <div class="path-arrow">→</div>
                <div class="path-node active">
                    <div class="node-title">Output</div>
                    <div class="node-value">705.6 kHz</div>
                </div>
            </div>
        </div>
    </main>
</body>
</html>
"""


def get_rtp_html() -> str:
    """Return RTP settings page HTML"""
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Magic Box - RTP設定</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            display: flex;
            min-height: 100vh;
        }

        /* Sidebar */
        .sidebar {
            width: 240px;
            background: #0f1419;
            border-right: 1px solid #1e2a3a;
            display: flex;
            flex-direction: column;
            padding: 20px 0;
        }
        .logo {
            padding: 0 20px 30px;
            border-bottom: 1px solid #1e2a3a;
            margin-bottom: 20px;
        }
        .logo h1 {
            color: #00d4ff;
            font-size: 1.4em;
            margin-bottom: 4px;
        }
        .logo .tagline {
            color: #666;
            font-size: 11px;
        }
        .nav-item {
            padding: 12px 20px;
            color: #aaa;
            text-decoration: none;
            display: block;
            transition: all 0.2s;
            border-left: 3px solid transparent;
        }
        .nav-item:hover {
            background: #1a2332;
            color: #00d4ff;
        }
        .nav-item.active {
            background: #1a2332;
            color: #00d4ff;
            border-left-color: #00d4ff;
        }

        /* Main content */
        .main-content {
            flex: 1;
            padding: 40px;
            overflow-y: auto;
        }
        .page-header {
            margin-bottom: 30px;
        }
        .page-header h2 {
            font-size: 2em;
            margin-bottom: 8px;
        }
        .page-header .subtitle {
            color: #888;
            font-size: 14px;
        }

        /* RTP status */
        .rtp-status {
            background: rgba(22, 33, 62, 0.6);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(0, 212, 255, 0.1);
        }
        .rtp-status h3 {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 16px;
            letter-spacing: 1px;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }
        .status-item {
            background: #0f3460;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        .status-item .label {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .status-item .value {
            font-size: 18px;
            font-weight: 600;
            color: #00d4ff;
        }
        .status-item.ok .value { color: #00ff88; }

        /* Settings section */
        .settings-section {
            background: rgba(22, 33, 62, 0.6);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(0, 212, 255, 0.1);
        }
        .settings-section h3 {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 20px;
            letter-spacing: 1px;
        }

        /* Form group */
        .form-group {
            background: #0f3460;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 12px;
        }
        .form-group label {
            display: block;
            font-size: 13px;
            color: #aaa;
            margin-bottom: 8px;
        }
        .form-group input[type="text"],
        .form-group input[type="number"],
        .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #1e2a3a;
            border-radius: 6px;
            background: #1a2332;
            color: #eee;
            font-size: 14px;
        }
        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: #00d4ff;
        }
        .form-group .hint {
            font-size: 11px;
            color: #666;
            margin-top: 4px;
        }

        /* Session list */
        .session-list {
            background: rgba(22, 33, 62, 0.6);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(0, 212, 255, 0.1);
        }
        .session-list h3 {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 16px;
            letter-spacing: 1px;
        }
        .session-item {
            background: #0f3460;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .session-info .session-name {
            font-size: 14px;
            font-weight: 600;
            color: #00d4ff;
            margin-bottom: 4px;
        }
        .session-info .session-details {
            font-size: 11px;
            color: #888;
        }
        .session-badge {
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        .session-badge.active {
            background: rgba(0, 255, 136, 0.2);
            color: #00ff88;
        }
        .session-badge.inactive {
            background: rgba(255, 68, 68, 0.2);
            color: #ff4444;
        }

        /* Action buttons */
        .action-buttons {
            display: flex;
            gap: 12px;
        }
        .action-buttons button {
            flex: 1;
            padding: 14px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary {
            background: #00ff88;
            color: #000;
        }
        .btn-primary:hover {
            background: #00cc6a;
        }
        .btn-secondary {
            background: #0f3460;
            color: #eee;
        }
        .btn-secondary:hover {
            background: #1a4b7c;
        }
    </style>
</head>
<body>
    <aside class="sidebar">
        <div class="logo">
            <h1>Magic Box</h1>
            <div class="tagline">Ultimate Audio Experience</div>
        </div>
        <nav>
            <a href="/mockup/dashboard" class="nav-item">📊 ダッシュボード</a>
            <a href="/mockup/headphones" class="nav-item">🎧 ヘッドホン選択</a>
            <a href="/mockup/eq" class="nav-item">🎚️ EQ設定</a>
            <a href="/mockup/system" class="nav-item">⚙️ システム設定</a>
            <a href="/mockup/status" class="nav-item">📈 ステータス監視</a>
            <a href="/mockup/rtp" class="nav-item active">🌐 RTP設定</a>
        </nav>
    </aside>

    <main class="main-content">
        <header class="page-header">
            <h2>RTP設定</h2>
            <p class="subtitle">ネットワークオーディオストリーミング設定</p>
        </header>

        <div class="rtp-status">
            <h3>RTP Receiver Status</h3>
            <div class="status-grid">
                <div class="status-item ok">
                    <div class="label">接続状態</div>
                    <div class="value">接続中</div>
                </div>

                <div class="status-item">
                    <div class="label">セッション数</div>
                    <div class="value">2 Active</div>
                </div>

                <div class="status-item">
                    <div class="label">受信レート</div>
                    <div class="value">48 kHz</div>
                </div>

                <div class="status-item ok">
                    <div class="label">パケットロス</div>
                    <div class="value">0.00%</div>
                </div>
            </div>
        </div>

        <div class="settings-section">
            <h3>Receiver Settings</h3>

            <div class="form-group">
                <label>Listen Address</label>
                <input type="text" value="0.0.0.0" placeholder="0.0.0.0">
                <div class="hint">すべてのインターフェースで待ち受け</div>
            </div>

            <div class="form-group">
                <label>Port Range</label>
                <input type="text" value="5004-5006" placeholder="5004-5006">
                <div class="hint">RTPストリームを受信するポート範囲</div>
            </div>

            <div class="form-group">
                <label>Buffer Size</label>
                <select>
                    <option>128 samples (低レイテンシ)</option>
                    <option selected>256 samples (推奨)</option>
                    <option>512 samples (安定重視)</option>
                    <option>1024 samples (最大安定性)</option>
                </select>
                <div class="hint">ネットワーク環境に応じて調整</div>
            </div>

            <div class="action-buttons">
                <button class="btn-secondary">デフォルトに戻す</button>
                <button class="btn-primary">適用</button>
            </div>
        </div>

        <div class="session-list">
            <h3>Active RTP Sessions</h3>

            <div class="session-item">
                <div class="session-info">
                    <div class="session-name">Raspberry Pi 4 Sender</div>
                    <div class="session-details">192.168.1.100:5004 → L24/48000/2</div>
                </div>
                <span class="session-badge active">Active</span>
            </div>

            <div class="session-item">
                <div class="session-info">
                    <div class="session-name">Desktop Sender</div>
                    <div class="session-details">192.168.1.50:5005 → L24/44100/2</div>
                </div>
                <span class="session-badge active">Active</span>
            </div>

            <div class="session-item">
                <div class="session-info">
                    <div class="session-name">Old Connection</div>
                    <div class="session-details">192.168.1.200:5006 → Timeout</div>
                </div>
                <span class="session-badge inactive">Inactive</span>
            </div>
        </div>
    </main>
</body>
</html>
"""
