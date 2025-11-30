"""Shared styles for Magic Box UI mockup - Mechanical/HUD design theme."""


def get_common_styles() -> str:
    """Return common CSS styles with mechanical/HUD theme"""
    return """
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Courier New', 'SF Mono', 'Consolas', monospace;
            background: #0a0e12;
            background-image:
                linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
            background-size: 20px 20px;
            color: #e0e0e0;
            display: flex;
            min-height: 100vh;
            position: relative;
        }

        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 50% 50%, rgba(0, 255, 255, 0.05) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }

        /* Sidebar */
        .sidebar {
            width: 240px;
            background: rgba(10, 14, 18, 0.95);
            border-right: 1px solid rgba(0, 255, 255, 0.2);
            box-shadow:
                inset -1px 0 2px rgba(0, 255, 255, 0.1),
                2px 0 20px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            padding: 20px 0;
            position: relative;
            z-index: 10;
        }
        .logo {
            padding: 0 20px 30px;
            border-bottom: 1px solid rgba(0, 255, 255, 0.2);
            margin-bottom: 20px;
            position: relative;
        }
        .logo h1 {
            color: #00ffff;
            font-size: 1.5em;
            margin-bottom: 4px;
            font-weight: 700;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
            letter-spacing: 2px;
        }
        .logo .tagline {
            color: #00ff88;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.8;
        }
        .nav-item {
            padding: 12px 20px;
            color: #80a0a0;
            text-decoration: none;
            display: block;
            transition: all 0.2s;
            border-left: 2px solid transparent;
            font-size: 13px;
            position: relative;
        }
        .nav-item:hover {
            background: rgba(0, 255, 255, 0.05);
            color: #00ffff;
            border-left-color: #00ffff;
            box-shadow: inset 0 0 20px rgba(0, 255, 255, 0.1);
        }
        .nav-item.active {
            background: rgba(0, 255, 255, 0.1);
            color: #00ffff;
            border-left-color: #00ffff;
            box-shadow:
                inset 0 0 20px rgba(0, 255, 255, 0.15),
                inset 2px 0 5px rgba(0, 255, 255, 0.3);
        }

        /* Main content */
        .main-content {
            flex: 1;
            padding: 40px;
            overflow-y: auto;
            position: relative;
            z-index: 1;
        }
        .page-header {
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(0, 255, 255, 0.2);
        }
        .page-header h2 {
            font-size: 2.2em;
            margin-bottom: 8px;
            color: #00ffff;
            text-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
            font-weight: 700;
            letter-spacing: 1px;
        }
        .page-header .subtitle {
            color: #00ff88;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.8;
        }

        /* Cards */
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background: rgba(18, 22, 26, 0.9);
            border-radius: 4px;
            padding: 20px;
            border: 1px solid rgba(0, 255, 255, 0.3);
            box-shadow:
                0 0 10px rgba(0, 255, 255, 0.1),
                inset 0 0 20px rgba(0, 255, 255, 0.02);
            position: relative;
            transition: all 0.3s;
        }
        .status-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.5), transparent);
        }
        .status-card:hover {
            border-color: rgba(0, 255, 255, 0.6);
            box-shadow:
                0 0 20px rgba(0, 255, 255, 0.2),
                inset 0 0 30px rgba(0, 255, 255, 0.05);
            transform: translateY(-2px);
        }
        .status-card .card-title {
            font-size: 11px;
            color: #00ff88;
            text-transform: uppercase;
            margin-bottom: 12px;
            letter-spacing: 2px;
            opacity: 0.8;
        }
        .status-card .card-value {
            font-size: 32px;
            font-weight: 700;
            color: #00ffff;
            margin-bottom: 8px;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
            font-family: 'Courier New', monospace;
        }
        .status-card .card-subtitle {
            font-size: 12px;
            color: #80a0a0;
        }
        .status-card.ok .card-value {
            color: #00ff88;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
        .status-card.warning .card-value {
            color: #ffaa00;
            text-shadow: 0 0 10px rgba(255, 170, 0, 0.5);
        }

        /* Quick action buttons */
        .quick-actions {
            background: rgba(18, 22, 26, 0.9);
            border-radius: 4px;
            padding: 24px;
            border: 1px solid rgba(0, 255, 255, 0.3);
            box-shadow:
                0 0 10px rgba(0, 255, 255, 0.1),
                inset 0 0 20px rgba(0, 255, 255, 0.02);
        }
        .quick-actions h3 {
            font-size: 13px;
            color: #00ff88;
            text-transform: uppercase;
            margin-bottom: 20px;
            letter-spacing: 2px;
        }
        .action-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }
        .action-btn {
            padding: 24px 20px;
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 4px;
            background: rgba(10, 14, 18, 0.8);
            color: #e0e0e0;
            text-decoration: none;
            display: block;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }
        .action-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.5), transparent);
        }
        .action-btn:hover {
            background: rgba(0, 255, 255, 0.05);
            border-color: #00ffff;
            box-shadow:
                0 0 20px rgba(0, 255, 255, 0.2),
                inset 0 0 30px rgba(0, 255, 255, 0.05);
            transform: translateY(-4px);
        }
        .action-btn .icon {
            font-size: 28px;
            margin-bottom: 12px;
            filter: drop-shadow(0 0 5px rgba(0, 255, 255, 0.3));
        }
        .action-btn .title {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 6px;
            color: #00ffff;
        }
        .action-btn .desc {
            font-size: 11px;
            color: #80a0a0;
        }
"""
