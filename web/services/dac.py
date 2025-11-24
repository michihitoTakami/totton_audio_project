"""DAC capability detection via ALSA."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Standard sample rates to test
STANDARD_RATES = [
    44100,
    48000,
    88200,
    96000,
    176400,
    192000,
    352800,
    384000,
    705600,
    768000,
]


@dataclass
class DacCapability:
    """DAC capability information."""

    device_name: str
    min_sample_rate: int
    max_sample_rate: int
    supported_rates: list[int]
    max_channels: int
    is_valid: bool
    error_message: str | None = None


def _parse_device_name(device: str) -> tuple[int | None, int | None]:
    """
    Parse ALSA device name to extract card and device numbers.

    Examples:
        "hw:0" -> (0, None)
        "hw:0,0" -> (0, 0)
        "plughw:1,0" -> (1, 0)
        "default" -> (None, None)
    """
    # Match hw:N or hw:N,M patterns
    match = re.match(r"(?:plug)?hw:(\d+)(?:,(\d+))?", device)
    if match:
        card = int(match.group(1))
        dev = int(match.group(2)) if match.group(2) else None
        return card, dev
    return None, None


def _scan_from_proc(card_num: int) -> DacCapability | None:
    """
    Scan DAC capabilities from /proc/asound/cardN/stream0.

    This method reads the USB Audio stream information directly.
    """
    stream_path = Path(f"/proc/asound/card{card_num}/stream0")
    if not stream_path.exists():
        return None

    try:
        content = stream_path.read_text()
    except OSError:
        return None

    # Parse Rates line
    # Format: "    Rates: 44100, 48000, 88200, 96000, ..."
    rates: list[int] = []
    channels = 2  # Default

    for line in content.split("\n"):
        line = line.strip()

        if line.startswith("Rates:"):
            rate_str = line[6:].strip()
            for part in rate_str.split(","):
                part = part.strip()
                try:
                    rates.append(int(part))
                except ValueError:
                    pass

        elif line.startswith("Channels:"):
            try:
                channels = int(line[9:].strip())
            except ValueError:
                pass

    if not rates:
        return None

    # Remove duplicates and sort
    rates = sorted(set(rates))

    return DacCapability(
        device_name=f"hw:{card_num}",
        min_sample_rate=min(rates),
        max_sample_rate=max(rates),
        supported_rates=rates,
        max_channels=channels,
        is_valid=True,
        error_message=None,
    )


def scan_dac_capability(device: str) -> DacCapability:
    """
    Scan DAC capabilities via ALSA.

    First tries to read from /proc/asound/cardN/stream0 (fastest, works for USB DACs).
    Falls back to aplay --dump-hw-params if /proc method fails.

    Args:
        device: ALSA device name (e.g., "hw:0", "hw:0,0", "default")

    Returns:
        DacCapability with supported rates and other info
    """
    # Try to parse card number from device name
    card_num, _ = _parse_device_name(device)

    # Method 1: Try /proc/asound/cardN/stream0 (for USB Audio devices)
    if card_num is not None:
        proc_cap = _scan_from_proc(card_num)
        if proc_cap is not None:
            proc_cap.device_name = device  # Use original device name
            return proc_cap

    # Method 2: Fall back to aplay --dump-hw-params
    return _scan_via_aplay(device)


def _scan_via_aplay(device: str) -> DacCapability:
    """
    Scan DAC capabilities using aplay --dump-hw-params.

    This is the fallback method for non-USB devices.
    """
    cap = DacCapability(
        device_name=device,
        min_sample_rate=0,
        max_sample_rate=0,
        supported_rates=[],
        max_channels=0,
        is_valid=False,
        error_message=None,
    )

    try:
        # Use aplay --dump-hw-params to get hardware parameters
        # This opens the device briefly and dumps all hardware capabilities
        result = subprocess.run(
            ["aplay", "--dump-hw-params", "-D", device, "/dev/null"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # aplay returns error but still outputs hw params to stderr
        output = result.stderr if result.stderr else result.stdout

        if not output:
            cap.error_message = "No output from aplay"
            return cap

        # Parse the output
        rate_min = None
        rate_max = None
        channels_max = None
        rate_values: list[int] = []

        for line in output.split("\n"):
            line = line.strip()

            # Parse RATE line
            # Format: "RATE: [44100 768000]" (range) or "RATE: 44100 48000 96000" (list)
            if line.startswith("RATE:"):
                rate_part = line[5:].strip()

                # Check if it's a range [min max]
                if rate_part.startswith("[") and "]" in rate_part:
                    range_str = rate_part[1 : rate_part.index("]")]
                    parts = range_str.split()
                    if len(parts) >= 2:
                        try:
                            rate_min = int(parts[0])
                            rate_max = int(parts[1])
                        except ValueError:
                            pass
                else:
                    # It's a list of values
                    parts = rate_part.split()
                    for p in parts:
                        try:
                            rate_values.append(int(p))
                        except ValueError:
                            pass

            # Parse CHANNELS line
            # Format: "CHANNELS: [1 2]" or "CHANNELS: 2"
            elif line.startswith("CHANNELS:"):
                channels_part = line[9:].strip()
                if channels_part.startswith("[") and "]" in channels_part:
                    range_str = channels_part[1 : channels_part.index("]")]
                    parts = range_str.split()
                    if len(parts) >= 2:
                        try:
                            channels_max = int(parts[-1])
                        except ValueError:
                            pass
                else:
                    parts = channels_part.split()
                    if parts:
                        try:
                            channels_max = int(parts[-1])
                        except ValueError:
                            pass

        # Determine supported rates
        if rate_values:
            # DAC reports specific supported rates
            cap.supported_rates = sorted(rate_values)
            cap.min_sample_rate = min(rate_values)
            cap.max_sample_rate = max(rate_values)
        elif rate_min is not None and rate_max is not None:
            # DAC reports a range - test standard rates within range
            cap.min_sample_rate = rate_min
            cap.max_sample_rate = rate_max
            cap.supported_rates = [
                r for r in STANDARD_RATES if rate_min <= r <= rate_max
            ]
        else:
            cap.error_message = "Could not parse sample rate information"
            return cap

        if channels_max:
            cap.max_channels = channels_max

        cap.is_valid = True
        return cap

    except subprocess.TimeoutExpired:
        cap.error_message = "Timeout while scanning device"
        return cap
    except FileNotFoundError:
        cap.error_message = "aplay command not found"
        return cap
    except subprocess.SubprocessError as e:
        cap.error_message = f"Subprocess error: {e}"
        return cap


def get_supported_output_rates(device: str, input_family: str = "44k") -> list[int]:
    """
    Get supported output rates for a given input family.

    Args:
        device: ALSA device name
        input_family: "44k" or "48k"

    Returns:
        List of supported output rates for the family
    """
    cap = scan_dac_capability(device)
    if not cap.is_valid:
        return []

    if input_family == "44k":
        family_rates = [44100, 88200, 176400, 352800, 705600]
    else:  # 48k
        family_rates = [48000, 96000, 192000, 384000, 768000]

    return [r for r in family_rates if r in cap.supported_rates]


def get_max_upsample_ratio(device: str, input_rate: int) -> int:
    """
    Get maximum supported upsampling ratio for a given input rate.

    Args:
        device: ALSA device name
        input_rate: Input sample rate (e.g., 44100, 48000)

    Returns:
        Maximum upsampling ratio (1, 2, 4, 8, or 16), or 0 if device is invalid
    """
    cap = scan_dac_capability(device)
    if not cap.is_valid:
        return 0

    # Try ratios from highest to lowest
    for ratio in [16, 8, 4, 2, 1]:
        target_rate = input_rate * ratio
        if target_rate in cap.supported_rates:
            return ratio

    # If no exact match, check if any supported rate >= input_rate
    for rate in sorted(cap.supported_rates, reverse=True):
        if rate >= input_rate and rate % input_rate == 0:
            return rate // input_rate

    return 0
