from __future__ import annotations


def test_resolve_any_interface_in_subnet_picks_matching_ip():
    from raspberry_pi import control_api

    mapping = {
        "eth0": "10.0.0.2",
        "enx1234": "192.168.55.100",
        "lo": "127.0.0.1",
    }

    def resolver(name: str):
        return mapping.get(name)

    host = control_api._resolve_any_interface_in_subnet(
        "192.168.55.0/24",
        interface_names=["lo", "eth0", "enx1234"],
        resolver=resolver,
    )
    assert host == "192.168.55.100"


def test_resolve_any_interface_in_subnet_returns_none_when_no_match():
    from raspberry_pi import control_api

    mapping = {
        "eth0": "10.0.0.2",
        "wlan0": "192.168.1.10",
    }

    def resolver(name: str):
        return mapping.get(name)

    host = control_api._resolve_any_interface_in_subnet(
        "192.168.55.0/24",
        interface_names=["eth0", "wlan0"],
        resolver=resolver,
    )
    assert host is None


def test_resolve_any_interface_in_subnet_skips_invalid_ips():
    from raspberry_pi import control_api

    mapping = {
        "enx1": "not-an-ip",
        "enx2": "192.168.55.101",
    }

    def resolver(name: str):
        return mapping.get(name)

    host = control_api._resolve_any_interface_in_subnet(
        "192.168.55.0/24",
        interface_names=["enx1", "enx2"],
        resolver=resolver,
    )
    assert host == "192.168.55.101"


def test_resolve_any_interface_in_subnet_invalid_subnet_returns_none():
    from raspberry_pi import control_api

    def resolver(_: str):
        return "192.168.55.100"

    host = control_api._resolve_any_interface_in_subnet(
        "not-a-cidr",
        interface_names=["enx0"],
        resolver=resolver,
    )
    assert host is None
