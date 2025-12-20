from __future__ import annotations

import sys


def test_control_api_import_does_not_import_control_plane():
    # `raspi-control-api` コンテナでは zmq が無いことがあるため、
    # control_api の import だけで control_plane(zmq) を要求しないことを保証する。
    assert "raspberry_pi.usb_i2s_bridge.control_plane" not in sys.modules
    __import__("raspberry_pi.control_api")
    assert "raspberry_pi.usb_i2s_bridge.control_plane" not in sys.modules
