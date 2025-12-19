"""ALSA device discovery and management."""

import subprocess


def get_alsa_devices() -> list[dict]:
    """
    Get list of available ALSA playback devices.

    Returns list of dicts with keys: id, name, description
    """
    devices: list[dict] = []
    seen: set[str] = set()

    def add_device(device_id: str, name: str, description: str) -> None:
        if not device_id or device_id in seen:
            return
        seen.add(device_id)
        devices.append({"id": device_id, "name": name, "description": description})

    # Get card list
    try:
        result = subprocess.run(
            ["aplay", "-l"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Parse output
            for line in result.stdout.split("\n"):
                if not line.startswith("card "):
                    continue
                # Parse: "card 0: PCH [HDA Intel PCH], device 0: ALC897 Analog..."
                parts = line.split(":")
                if len(parts) < 2:
                    continue
                card_part = parts[0]  # "card 0"
                card_num = card_part.split()[1]

                # Card short id (e.g. "USB", "PCH") comes after ":"
                card_id = parts[1].strip().split()[0] if parts[1].strip() else ""

                # Get device number
                device_match = line.find("device ")
                if device_match == -1:
                    continue
                device_part = line[device_match:].split(":")[0]
                device_num = device_part.split()[1]

                # Extract card display name (prefer [...] content)
                name_start = parts[1].find("[")
                name_end = parts[1].find("]")
                if name_start != -1 and name_end != -1:
                    card_name = parts[1][name_start + 1 : name_end].strip()
                else:
                    card_name = parts[1].strip().split(",")[0].strip()

                description = line.strip()

                # Prefer showing card id + display name
                display = card_name or card_id or f"card {card_num}"
                if card_id and card_name and card_id not in card_name:
                    display = f"{card_id} ({card_name})"

                # Canonical numeric device id (always available)
                add_device(f"hw:{card_num},{device_num}", display, description)

                # Add card-id based aliases when available (helps users who use hw:USB, etc.)
                if card_id:
                    add_device(f"hw:{card_id},{device_num}", display, description)
                    if device_num == "0":
                        add_device(f"hw:{card_id}", display, description)
                        add_device(f"hw:{card_num}", display, description)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Always include default device
    default = {"id": "default", "name": "Default", "description": "System default"}
    if "default" not in seen:
        devices.insert(0, default)

    return devices
