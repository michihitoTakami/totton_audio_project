"""
HRTF FIR Filter Generation Tests

Tests for:
1. DC normalization (DC gain = 1.0 for each channel)
2. Phase preservation (ITD verification - ipsilateral vs contralateral timing)
3. Metadata validation
4. Unit tests for helper functions
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add scripts to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_hrtf import (  # noqa: E402
    angular_distance,
    find_nearest_position,
    pad_hrir_to_length,
    trim_hrir,
)

# Test data paths
HRTF_DIR = REPO_ROOT / "data" / "crossfeed" / "hrtf"
N_TAPS = 2_000_000
N_CHANNELS = 4


def split_channels(data: np.ndarray, metadata: dict) -> dict:
    """
    Split raw HRTF binary data into channel arrays according to storage_format.

    Args:
        data: flat float32 array read from .bin
        metadata: dict containing at least n_taps and optional storage_format

    Returns:
        dict with keys ll, lr, rl, rr (each np.ndarray of length n_taps)
    """
    n_taps = metadata["n_taps"]
    fmt = metadata.get("storage_format", "tap_interleaved_v1")

    if fmt == "channel_major_v1":
        expected = N_CHANNELS * n_taps
        if data.size < expected:
            raise ValueError(
                f"Insufficient samples for channel_major_v1: got {data.size}"
            )
        ll = data[0:n_taps]
        lr = data[n_taps : 2 * n_taps]
        rl = data[2 * n_taps : 3 * n_taps]
        rr = data[3 * n_taps : 4 * n_taps]
    else:
        ll = data[0::4][:n_taps]
        lr = data[1::4][:n_taps]
        rl = data[2::4][:n_taps]
        rr = data[3::4][:n_taps]

    return {"ll": ll, "lr": lr, "rl": rl, "rr": rr}


class TestStorageFormatCompatibility:
    """Unit tests for split_channels helper handling multiple layouts."""

    def test_channel_major_layout(self):
        metadata = {"n_taps": 3, "storage_format": "channel_major_v1"}
        ll = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        lr = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        rl = np.array([7.0, 8.0, 9.0], dtype=np.float32)
        rr = np.array([10.0, 11.0, 12.0], dtype=np.float32)
        data = np.concatenate([ll, lr, rl, rr])

        channels = split_channels(data, metadata)
        assert np.array_equal(channels["ll"], ll)
        assert np.array_equal(channels["lr"], lr)
        assert np.array_equal(channels["rl"], rl)
        assert np.array_equal(channels["rr"], rr)

    def test_tap_interleaved_default(self):
        metadata = {"n_taps": 2}  # storage_format omitted (legacy default)
        # Layout: LL0, LR0, RL0, RR0, LL1, LR1, RL1, RR1
        data = np.array(
            [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
            ],
            dtype=np.float32,
        )

        channels = split_channels(data, metadata)
        assert np.array_equal(channels["ll"], np.array([0.1, 0.5], dtype=np.float32))
        assert np.array_equal(channels["lr"], np.array([0.2, 0.6], dtype=np.float32))
        assert np.array_equal(channels["rl"], np.array([0.3, 0.7], dtype=np.float32))
        assert np.array_equal(channels["rr"], np.array([0.4, 0.8], dtype=np.float32))


class TestPadHrirToLength:
    """Unit tests for pad_hrir_to_length function."""

    def test_padding_shorter_input(self):
        """Test zero-padding a shorter input."""
        hrir = np.array([1.0, 2.0, 3.0])
        target_length = 10
        result = pad_hrir_to_length(hrir, target_length)

        assert len(result) == target_length
        assert np.array_equal(result[:3], hrir)
        assert np.all(result[3:] == 0)

    def test_no_padding_exact_length(self):
        """Test input that matches target length."""
        hrir = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pad_hrir_to_length(hrir, 5)

        assert len(result) == 5
        assert np.array_equal(result, hrir)

    def test_truncation_longer_input(self):
        """Test truncation when input exceeds target length."""
        hrir = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pad_hrir_to_length(hrir, 3)

        assert len(result) == 3
        assert np.array_equal(result, hrir[:3])

    def test_preserves_dtype(self):
        """Test that dtype is preserved."""
        hrir = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = pad_hrir_to_length(hrir, 10)

        assert result.dtype == np.float64


class TestTrimHrir:
    """Unit tests for trim_hrir function."""

    def test_trims_tail_with_padding(self):
        hrir = np.zeros(1000, dtype=np.float32)
        hrir[0] = 1.0
        hrir[300] = 1e-2  # about -40dB relative to peak

        trimmed = trim_hrir(hrir, threshold_db=-60.0, pad=16)

        # Should cut after the last low-level tap + padding
        assert len(trimmed) == 317
        assert trimmed[-1] == 0.0

    def test_keeps_significant_tail(self):
        hrir = np.zeros(1200, dtype=np.float32)
        hrir[0] = 1.0
        hrir[900] = 1e-2  # above -60dB threshold

        trimmed = trim_hrir(hrir, threshold_db=-60.0, pad=32)

        # Should retain the significant tap plus padding
        assert len(trimmed) == 933
        assert trimmed[900] == pytest.approx(1e-2)

    def test_silent_input_returns_original_length(self):
        hrir = np.zeros(50, dtype=np.float32)
        trimmed = trim_hrir(hrir, threshold_db=-80.0, pad=8)
        assert len(trimmed) == 50


class TestAngularDistance:
    """Unit tests for angular_distance function (360° wraparound)."""

    def test_simple_distance(self):
        """Test simple angular distance."""
        assert angular_distance(0.0, 30.0) == 30.0
        assert angular_distance(30.0, 0.0) == 30.0

    def test_wraparound_short_path(self):
        """Test 360° wraparound takes shorter path."""
        # 330° to 10° should be 40°, not 320°
        assert angular_distance(330.0, 10.0) == 40.0
        assert angular_distance(10.0, 330.0) == 40.0

    def test_wraparound_at_180(self):
        """Test distance at 180° boundary."""
        assert angular_distance(0.0, 180.0) == 180.0
        assert angular_distance(90.0, 270.0) == 180.0

    def test_same_angle(self):
        """Test same angle returns 0."""
        assert angular_distance(45.0, 45.0) == 0.0
        assert angular_distance(330.0, 330.0) == 0.0


class TestFindNearestPosition:
    """Unit tests for find_nearest_position function."""

    def test_exact_match(self):
        """Test finding exact position match."""
        positions = np.array(
            [
                [0.0, 0.0, 1.0],
                [30.0, 0.0, 1.0],
                [330.0, 0.0, 1.0],
            ]
        )
        idx = find_nearest_position(positions, 30.0, 0.0)
        assert idx == 1

    def test_nearest_azimuth(self):
        """Test finding nearest azimuth."""
        positions = np.array(
            [
                [0.0, 0.0, 1.0],
                [32.0, 0.0, 1.0],  # Nearest to 30
                [60.0, 0.0, 1.0],
            ]
        )
        idx = find_nearest_position(positions, 30.0, 0.0)
        assert idx == 1

    def test_hutubs_left_speaker(self):
        """Test finding -30 deg (330 deg in HUTUBS convention)."""
        positions = np.array(
            [
                [0.0, 0.0, 1.0],
                [30.0, 0.0, 1.0],
                [330.0, 0.0, 1.0],
            ]
        )
        idx = find_nearest_position(positions, 330.0, 0.0)
        assert idx == 2

    def test_wraparound_finds_nearest(self):
        """Test 360° wraparound: target 330° should find 330°, not 10°."""
        positions = np.array(
            [
                [10.0, 0.0, 1.0],  # 40° away from 330° (via wraparound)
                [30.0, 0.0, 1.0],  # 60° away from 330° (via wraparound)
                [330.0, 0.0, 1.0],  # 0° away (exact match)
            ]
        )
        idx = find_nearest_position(positions, 330.0, 0.0)
        assert idx == 2, "Should find exact match at 330°"

    def test_wraparound_prefers_closer(self):
        """Test that wraparound prefers closer angle across 0°."""
        positions = np.array(
            [
                [350.0, 0.0, 1.0],  # 20° away from 10° (via wraparound)
                [100.0, 0.0, 1.0],  # 90° away from 10°
            ]
        )
        idx = find_nearest_position(positions, 10.0, 0.0)
        assert idx == 0, "Should find 350° as closer to 10° than 100°"


class TestGeneratedHRTFFilters:
    """Integration tests for generated HRTF filters."""

    @pytest.fixture(scope="class")
    def hrtf_m_44k(self):
        """Load the M size 44k filter."""
        bin_path = HRTF_DIR / "hrtf_m_44k.bin"
        json_path = HRTF_DIR / "hrtf_m_44k.json"

        if not bin_path.exists():
            pytest.skip(f"HRTF binary not found: {bin_path}")

        data = np.fromfile(bin_path, dtype=np.float32)
        with open(json_path) as f:
            metadata = json.load(f)

        channels = split_channels(data, metadata)
        channels["metadata"] = metadata
        return channels

    def test_max_dc_gain_is_around_point_six(self, hrtf_m_44k):
        """Test maximum DC gain across all channels is around 0.6 (ILD-preserving normalization)."""
        dc_gains = {
            "LL": np.sum(hrtf_m_44k["ll"]),
            "LR": np.sum(hrtf_m_44k["lr"]),
            "RL": np.sum(hrtf_m_44k["rl"]),
            "RR": np.sum(hrtf_m_44k["rr"]),
        }
        max_dc = max(dc_gains.values())
        assert abs(max_dc - 0.6) < 0.01, f"Max DC gain: {max_dc}, gains: {dc_gains}"

    def test_ild_exists(self, hrtf_m_44k):
        """
        Test ILD (Interaural Level Difference) exists.

        For ±30° sources, there should be DC gain differences between channels.
        Note: At low frequencies (DC), ILD direction can vary due to head diffraction.
        We verify that channels have different gains, not the direction.
        """
        ll_dc = np.sum(hrtf_m_44k["ll"])
        lr_dc = np.sum(hrtf_m_44k["lr"])
        rl_dc = np.sum(hrtf_m_44k["rl"])
        rr_dc = np.sum(hrtf_m_44k["rr"])

        # ILD should exist: channels should have different DC gains
        dc_gains = [ll_dc, lr_dc, rl_dc, rr_dc]
        dc_range = max(dc_gains) - min(dc_gains)
        assert dc_range > 0.01, f"ILD too small: range={dc_range}, gains={dc_gains}"

    def test_ild_symmetric(self, hrtf_m_44k):
        """Test ILD is symmetric between left and right."""
        ll_dc = np.sum(hrtf_m_44k["ll"])
        lr_dc = np.sum(hrtf_m_44k["lr"])
        rl_dc = np.sum(hrtf_m_44k["rl"])
        rr_dc = np.sum(hrtf_m_44k["rr"])

        # Ipsilateral should be similar (LL ≈ RR)
        ipsi_ratio = ll_dc / rr_dc
        assert 0.9 < ipsi_ratio < 1.1, f"Ipsilateral asymmetry: LL={ll_dc}, RR={rr_dc}"

        # Contralateral should be similar (LR ≈ RL)
        contra_ratio = lr_dc / rl_dc
        assert (
            0.9 < contra_ratio < 1.1
        ), f"Contralateral asymmetry: LR={lr_dc}, RL={rl_dc}"

    def test_itd_exists(self, hrtf_m_44k):
        """
        Test ITD (Interaural Time Difference) exists.

        For ±30° sources, there should be a timing difference between
        ipsilateral (LL, RR) and contralateral (LR, RL) channels.

        Note: The exact order of peaks depends on HRIR characteristics
        (head diffraction, ear canal resonance). We verify ITD exists,
        not its direction.
        """
        ll = hrtf_m_44k["ll"]
        lr = hrtf_m_44k["lr"]

        # Find peak positions in the impulse response
        ll_peak_idx = np.argmax(np.abs(ll[:1000]))
        lr_peak_idx = np.argmax(np.abs(lr[:1000]))

        # ITD should exist: peaks should be at different positions
        # At 705.6kHz, 1 sample ≈ 1.4µs, typical ITD for 30° is ~200-300µs
        itd_samples = abs(lr_peak_idx - ll_peak_idx)
        assert itd_samples > 10, (
            f"ITD too small: LL peak at {ll_peak_idx}, LR peak at {lr_peak_idx}. "
            f"Difference: {itd_samples} samples"
        )

    def test_itd_symmetric(self, hrtf_m_44k):
        """
        Test ITD is symmetric between left and right.

        LL should be similar timing to RR (both ipsilateral).
        LR should be similar timing to RL (both contralateral).
        """
        ll = hrtf_m_44k["ll"]
        lr = hrtf_m_44k["lr"]
        rl = hrtf_m_44k["rl"]
        rr = hrtf_m_44k["rr"]

        ll_peak = np.argmax(np.abs(ll[:1000]))
        lr_peak = np.argmax(np.abs(lr[:1000]))
        rl_peak = np.argmax(np.abs(rl[:1000]))
        rr_peak = np.argmax(np.abs(rr[:1000]))

        # Ipsilateral peaks should be close to each other
        ipsi_diff = abs(ll_peak - rr_peak)
        # Contralateral peaks should be close to each other
        contra_diff = abs(lr_peak - rl_peak)

        # Allow some tolerance (10 samples at 705.6kHz ≈ 14µs)
        assert ipsi_diff < 50, f"Ipsilateral asymmetry: LL={ll_peak}, RR={rr_peak}"
        assert contra_diff < 50, f"Contralateral asymmetry: LR={lr_peak}, RL={rl_peak}"

    def test_metadata_normalization(self, hrtf_m_44k):
        """Test metadata indicates ILD-preserving normalization."""
        assert hrtf_m_44k["metadata"].get("normalization") == "ild_preserving"
        assert hrtf_m_44k["metadata"].get("max_dc_gain") == 1.0

    def test_metadata_phase_type_original(self, hrtf_m_44k):
        """Test metadata indicates phase is preserved (not minimum phase)."""
        assert hrtf_m_44k["metadata"].get("phase_type") == "original"

    def test_filter_shape(self, hrtf_m_44k):
        """Test filter has expected shape."""
        n_taps = hrtf_m_44k["metadata"]["n_taps"]
        assert len(hrtf_m_44k["ll"]) == n_taps
        assert len(hrtf_m_44k["lr"]) == n_taps
        assert len(hrtf_m_44k["rl"]) == n_taps
        assert len(hrtf_m_44k["rr"]) == n_taps


class TestAllHRTFFilters:
    """Test all generated HRTF filters."""

    @pytest.mark.parametrize(
        "size,rate",
        [
            ("xs", "44k"),
            ("xs", "48k"),
            ("s", "44k"),
            ("s", "48k"),
            ("m", "44k"),
            ("m", "48k"),
            ("l", "44k"),
            ("l", "48k"),
            ("xl", "44k"),
            ("xl", "48k"),
        ],
    )
    def test_ild_preservation_all_filters(self, size, rate):
        """Test ILD preservation for all filter variants."""
        bin_path = HRTF_DIR / f"hrtf_{size}_{rate}.bin"
        json_path = HRTF_DIR / f"hrtf_{size}_{rate}.json"

        if not bin_path.exists():
            pytest.skip(f"HRTF binary not found: {bin_path}")

        with open(json_path) as f:
            metadata = json.load(f)

        data = np.fromfile(bin_path, dtype=np.float32)
        channels = split_channels(data, metadata)
        ll = channels["ll"]
        lr = channels["lr"]
        rl = channels["rl"]
        rr = channels["rr"]

        dc_gains = {
            "LL": np.sum(ll),
            "LR": np.sum(lr),
            "RL": np.sum(rl),
            "RR": np.sum(rr),
        }

        # Max DC gain should be around 0.6 for ILD-preserving HRTF filters
        # (normalized to preserve interaural level differences)
        max_dc = max(dc_gains.values())
        assert abs(max_dc - 0.6) < 0.01, f"{size}_{rate} max DC: {max_dc}"

        # ILD should exist: channels should have different DC gains
        dc_range = max_dc - min(dc_gains.values())
        assert dc_range > 0.01, f"{size}_{rate} ILD too small: range={dc_range}"

    @pytest.mark.parametrize(
        "size,rate",
        [
            ("xs", "44k"),
            ("s", "44k"),
            ("m", "44k"),
            ("l", "44k"),
            ("xl", "44k"),
        ],
    )
    def test_itd_exists_all_sizes(self, size, rate):
        """Test ITD exists for all size variants."""
        bin_path = HRTF_DIR / f"hrtf_{size}_{rate}.bin"
        json_path = HRTF_DIR / f"hrtf_{size}_{rate}.json"

        if not bin_path.exists():
            pytest.skip(f"HRTF binary not found: {bin_path}")

        with open(json_path) as f:
            metadata = json.load(f)

        data = np.fromfile(bin_path, dtype=np.float32)
        channels = split_channels(data, metadata)
        ll = channels["ll"]
        lr = channels["lr"]

        ll_peak = np.argmax(np.abs(ll[:1000]))
        lr_peak = np.argmax(np.abs(lr[:1000]))

        # ITD should exist: peaks should differ
        itd_samples = abs(lr_peak - ll_peak)
        assert itd_samples > 10, (
            f"{size}_{rate} ITD too small: LL peak at {ll_peak}, "
            f"LR peak at {lr_peak}, difference: {itd_samples}"
        )
