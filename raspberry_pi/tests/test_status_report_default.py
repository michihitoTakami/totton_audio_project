from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_status_report_url_falls_back_to_env_when_config_empty(
    monkeypatch, tmp_path: Path
) -> None:
    cfg_path = tmp_path / "config.env"
    cfg_path.write_text("USB_I2S_STATUS_REPORT_URL=\n")
    monkeypatch.setenv("USB_I2S_CONFIG_PATH", str(cfg_path))
    monkeypatch.setenv(
        "USB_I2S_STATUS_REPORT_URL", "http://example.local/i2s/peer-status"
    )

    bridge = importlib.import_module("raspberry_pi.usb_i2s_bridge.bridge")
    reloaded = importlib.reload(bridge)
    try:
        cfg = reloaded.UsbI2sBridgeConfig()
        assert cfg.status_report_url == "http://example.local/i2s/peer-status"
    finally:
        monkeypatch.delenv("USB_I2S_CONFIG_PATH", raising=False)
        monkeypatch.delenv("USB_I2S_STATUS_REPORT_URL", raising=False)
        importlib.reload(bridge)
