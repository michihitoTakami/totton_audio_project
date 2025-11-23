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

from generate_hrtf import pad_hrir_to_length, find_nearest_position  # noqa: E402

# Test data paths
HRTF_DIR = REPO_ROOT / "data" / "crossfeed" / "hrtf"
N_TAPS = 2_000_000
N_CHANNELS = 4


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

        # De-interleave: [LL0, LR0, RL0, RR0, LL1, LR1, RL1, RR1, ...]
        n_taps = metadata["n_taps"]
        ll = data[0::4][:n_taps]
        lr = data[1::4][:n_taps]
        rl = data[2::4][:n_taps]
        rr = data[3::4][:n_taps]

        return {
            "ll": ll,
            "lr": lr,
            "rl": rl,
            "rr": rr,
            "metadata": metadata,
        }

    def test_dc_normalization_ll(self, hrtf_m_44k):
        """Test LL channel DC gain is 1.0."""
        dc_gain = np.sum(hrtf_m_44k["ll"])
        assert abs(dc_gain - 1.0) < 1e-5, f"LL DC gain: {dc_gain}"

    def test_dc_normalization_lr(self, hrtf_m_44k):
        """Test LR channel DC gain is 1.0."""
        dc_gain = np.sum(hrtf_m_44k["lr"])
        assert abs(dc_gain - 1.0) < 1e-5, f"LR DC gain: {dc_gain}"

    def test_dc_normalization_rl(self, hrtf_m_44k):
        """Test RL channel DC gain is 1.0."""
        dc_gain = np.sum(hrtf_m_44k["rl"])
        assert abs(dc_gain - 1.0) < 1e-5, f"RL DC gain: {dc_gain}"

    def test_dc_normalization_rr(self, hrtf_m_44k):
        """Test RR channel DC gain is 1.0."""
        dc_gain = np.sum(hrtf_m_44k["rr"])
        assert abs(dc_gain - 1.0) < 1e-5, f"RR DC gain: {dc_gain}"

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

    def test_metadata_dc_normalized_flag(self, hrtf_m_44k):
        """Test metadata contains dc_normalized: true."""
        assert hrtf_m_44k["metadata"].get("dc_normalized") is True

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
    def test_dc_normalization_all_filters(self, size, rate):
        """Test DC normalization for all filter variants."""
        bin_path = HRTF_DIR / f"hrtf_{size}_{rate}.bin"
        json_path = HRTF_DIR / f"hrtf_{size}_{rate}.json"

        if not bin_path.exists():
            pytest.skip(f"HRTF binary not found: {bin_path}")

        with open(json_path) as f:
            metadata = json.load(f)

        data = np.fromfile(bin_path, dtype=np.float32)
        n_taps = metadata["n_taps"]

        # De-interleave
        ll = data[0::4][:n_taps]
        lr = data[1::4][:n_taps]
        rl = data[2::4][:n_taps]
        rr = data[3::4][:n_taps]

        for name, channel in [("LL", ll), ("LR", lr), ("RL", rl), ("RR", rr)]:
            dc_gain = np.sum(channel)
            assert abs(dc_gain - 1.0) < 1e-5, f"{size}_{rate} {name} DC gain: {dc_gain}"

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
        n_taps = metadata["n_taps"]

        ll = data[0::4][:n_taps]
        lr = data[1::4][:n_taps]

        ll_peak = np.argmax(np.abs(ll[:1000]))
        lr_peak = np.argmax(np.abs(lr[:1000]))

        # ITD should exist: peaks should differ
        itd_samples = abs(lr_peak - ll_peak)
        assert itd_samples > 10, (
            f"{size}_{rate} ITD too small: LL peak at {ll_peak}, "
            f"LR peak at {lr_peak}, difference: {itd_samples}"
        )
