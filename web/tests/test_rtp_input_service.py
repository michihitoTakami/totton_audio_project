import asyncio

import pytest

from web.models import RtpInputConfigUpdate, RtpInputSettings
from web.services.rtp_input import build_gst_command, RtpReceiverManager


def test_build_gst_command_supports_encodings():
    base = RtpInputSettings()
    l16 = build_gst_command(base.model_copy(update={"encoding": "L16"}))
    l24 = build_gst_command(base.model_copy(update={"encoding": "L24"}))
    l32 = build_gst_command(base.model_copy(update={"encoding": "L32"}))

    assert "rtpL16depay" in l16
    assert "rtpL24depay" in l24
    assert "rtpL32depay" in l32
    assert any("latency=100" in part for part in l24)
    assert "rtpbin" in l24
    assert any("rtcp" in part for part in l24)


def test_config_update_merges_and_validates():
    manager = RtpReceiverManager(settings=RtpInputSettings())
    updated = asyncio.run(
        manager.apply_config(
            RtpInputConfigUpdate(
                port=46000,
                latency_ms=250,
                rtcp_port=46010,
                rtcp_send_port=46011,
                sender_host="192.168.0.10",
            )
        )
    )
    assert updated.port == 46000
    assert updated.latency_ms == 250
    assert updated.rtcp_port == 46010
    assert updated.rtcp_send_port == 46011
    assert updated.sender_host == "192.168.0.10"

    with pytest.raises(Exception):
        asyncio.run(manager.apply_config(RtpInputConfigUpdate(latency_ms=1)))


class _DummyProcess:
    def __init__(self):
        self.returncode = None
        self.pid = 1234
        self._terminated = False

    def terminate(self):
        self._terminated = True

    async def wait(self):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


async def _dummy_runner(_cmd):
    return _DummyProcess()


def test_start_stop_uses_runner():
    manager = RtpReceiverManager(
        settings=RtpInputSettings(), process_runner=_dummy_runner
    )
    asyncio.run(manager.start())
    status = asyncio.run(manager.status())
    assert status.running is True
    assert status.pid == 1234

    asyncio.run(manager.stop())
    status_after = asyncio.run(manager.status())
    assert status_after.running is False
