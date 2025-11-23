"""ALSA device discovery and management."""

import subprocess


def get_alsa_devices() -> list[dict]:
    """
    Get list of available ALSA playback devices.

    Returns list of dicts with keys: id, name, description
    """
    devices: list[dict] = []

    # Get card list
    try:
        result = subprocess.run(
            ["aplay", "-l"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return devices

        # Parse output
        for line in result.stdout.split("\n"):
            if line.startswith("card "):
                # Parse: "card 0: PCH [HDA Intel PCH], device 0: ALC897 Analog..."
                parts = line.split(":")
                if len(parts) >= 2:
                    card_part = parts[0]  # "card 0"
                    card_num = card_part.split()[1]

                    # Get device number
                    device_match = line.find("device ")
                    if device_match != -1:
                        device_part = line[device_match:].split(":")[0]
                        device_num = device_part.split()[1]

                        # Extract name
                        name_start = parts[1].find("[")
                        name_end = parts[1].find("]")
                        if name_start != -1 and name_end != -1:
                            card_name = parts[1][name_start + 1 : name_end]
                        else:
                            card_name = parts[1].strip().split(",")[0]

                        device_id = f"hw:{card_num},{device_num}"
                        devices.append(
                            {
                                "id": device_id,
                                "name": card_name,
                                "description": line.strip(),
                            }
                        )
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Add default device
    devices.insert(
        0, {"id": "default", "name": "Default", "description": "System default"}
    )

    return devices
