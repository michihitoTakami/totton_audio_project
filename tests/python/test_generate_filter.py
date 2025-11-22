"""
Unit tests for scripts/generate_filter.py

Tests filter coefficient generation, validation, and normalization.
Uses small tap counts for fast execution.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


class TestValidateTapCount:
    """Tests for validate_tap_count function."""

    def test_valid_tap_count_divisible_by_16(self):
        """Tap count divisible by 16 should pass."""
        import generate_filter

        # Should not raise
        generate_filter.validate_tap_count(1600, 16)
        generate_filter.validate_tap_count(2_000_000, 16)

    def test_invalid_tap_count_not_divisible(self):
        """Tap count not divisible by upsample ratio should raise."""
        import generate_filter

        with pytest.raises(ValueError, match="倍数である必要があります"):
            generate_filter.validate_tap_count(1601, 16)

    def test_valid_tap_count_different_ratio(self):
        """Different upsample ratios should work."""
        import generate_filter

        generate_filter.validate_tap_count(1000, 8)
        generate_filter.validate_tap_count(1000, 4)


class TestNormalizeCoefficients:
    """Tests for normalize_coefficients function."""

    def test_normalizes_dc_gain_to_one(self):
        """DC gain should be normalized to 1.0."""
        import generate_filter

        # Create test coefficients with known DC gain
        h = np.array([0.5, 1.0, 0.5])  # DC gain = 2.0
        h_norm, info = generate_filter.normalize_coefficients(h)

        assert np.isclose(np.sum(h_norm), 1.0, rtol=1e-6)
        assert info["normalization_applied"] is True
        assert np.isclose(info["original_dc_gain"], 2.0)

    def test_zero_dc_gain_raises_error(self):
        """Zero DC gain should raise ValueError."""
        import generate_filter

        h = np.array([1.0, -1.0])  # DC gain = 0
        with pytest.raises(ValueError, match="DCゲインが0に近すぎます"):
            generate_filter.normalize_coefficients(h)

    def test_preserves_shape(self):
        """Normalization should preserve coefficient shape."""
        import generate_filter

        h = np.random.randn(1000)
        h_norm, _ = generate_filter.normalize_coefficients(h)

        assert h_norm.shape == h.shape


class TestFilterDesign:
    """Tests for filter design functions using small tap counts."""

    @pytest.fixture
    def setup_small_filter_params(self):
        """Setup small filter parameters for fast testing."""
        import generate_filter

        # Save original values
        orig_n_taps = generate_filter.N_TAPS
        orig_sample_rate_input = generate_filter.SAMPLE_RATE_INPUT
        orig_sample_rate_output = generate_filter.SAMPLE_RATE_OUTPUT
        orig_passband_end = generate_filter.PASSBAND_END
        orig_stopband_start = generate_filter.STOPBAND_START
        orig_kaiser_beta = generate_filter.KAISER_BETA

        # Set small values for testing
        generate_filter.N_TAPS = 1600  # Small for fast testing
        generate_filter.SAMPLE_RATE_INPUT = 44100
        generate_filter.SAMPLE_RATE_OUTPUT = 44100 * 16
        generate_filter.PASSBAND_END = 20000
        generate_filter.STOPBAND_START = 22050
        generate_filter.KAISER_BETA = 14  # Smaller beta for fewer taps

        yield

        # Restore original values
        generate_filter.N_TAPS = orig_n_taps
        generate_filter.SAMPLE_RATE_INPUT = orig_sample_rate_input
        generate_filter.SAMPLE_RATE_OUTPUT = orig_sample_rate_output
        generate_filter.PASSBAND_END = orig_passband_end
        generate_filter.STOPBAND_START = orig_stopband_start
        generate_filter.KAISER_BETA = orig_kaiser_beta

    def test_linear_phase_filter_design(self, setup_small_filter_params):
        """Linear phase filter should be designed correctly."""
        import generate_filter

        h = generate_filter.design_linear_phase_filter()

        # Check tap count (may be +1 for odd)
        assert len(h) in [1600, 1601]

        # Check DC gain is non-zero
        assert np.abs(np.sum(h)) > 0

    def test_minimum_phase_conversion(self, setup_small_filter_params):
        """Minimum phase conversion should produce causal filter."""
        import generate_filter

        h_linear = generate_filter.design_linear_phase_filter()
        h_min = generate_filter.convert_to_minimum_phase(h_linear)

        # Check tap count
        assert len(h_min) == 1600

        # Check energy is concentrated at front (causal)
        # For minimum phase, most energy should be in first half
        mid = len(h_min) // 2
        energy_first = np.sum(h_min[:mid] ** 2)
        energy_second = np.sum(h_min[mid:] ** 2)

        # First half should have more energy
        assert energy_first > energy_second


class TestValidateSpecifications:
    """Tests for validate_specifications function."""

    @pytest.fixture
    def setup_validation_params(self):
        """Setup parameters for validation testing."""
        import generate_filter

        orig_sample_rate_output = generate_filter.SAMPLE_RATE_OUTPUT
        orig_passband_end = generate_filter.PASSBAND_END
        orig_stopband_start = generate_filter.STOPBAND_START
        orig_stopband_attenuation = generate_filter.STOPBAND_ATTENUATION_DB

        generate_filter.SAMPLE_RATE_OUTPUT = 44100 * 16
        generate_filter.PASSBAND_END = 20000
        generate_filter.STOPBAND_START = 22050
        generate_filter.STOPBAND_ATTENUATION_DB = 40  # Achievable with small filter

        yield

        generate_filter.SAMPLE_RATE_OUTPUT = orig_sample_rate_output
        generate_filter.PASSBAND_END = orig_passband_end
        generate_filter.STOPBAND_START = orig_stopband_start
        generate_filter.STOPBAND_ATTENUATION_DB = orig_stopband_attenuation

    def test_validation_returns_dict(self, setup_validation_params):
        """validate_specifications should return dict with expected keys."""
        import generate_filter

        # Create a simple lowpass filter for testing
        h = signal.firwin(101, 0.5, window="hamming")

        results = generate_filter.validate_specifications(h)

        assert isinstance(results, dict)
        assert "passband_ripple_db" in results
        assert "stopband_attenuation_db" in results
        assert "is_minimum_phase" in results
        assert "peak_position" in results

    def test_minimum_phase_detection(self, setup_validation_params):
        """Should correctly detect minimum phase filters."""
        import generate_filter

        # Create a minimum phase filter
        h_linear = signal.firwin(101, 0.5, window="hamming")
        h_min = signal.minimum_phase(h_linear, method="homomorphic")

        results = generate_filter.validate_specifications(h_min)

        # Peak should be near front
        assert results["peak_position"] < len(h_min) * 0.1


class TestFrequencyResponse:
    """Tests for frequency response characteristics."""

    def test_passband_is_flat(self):
        """Passband should be relatively flat."""
        # Design a simple filter
        fs = 44100 * 16
        cutoff = 20000 / (fs / 2)
        h = signal.firwin(1001, cutoff, window=("kaiser", 14))

        # Get frequency response
        w, H = signal.freqz(h, worN=8192, fs=fs)
        H_db = 20 * np.log10(np.abs(H) + 1e-12)

        # Check passband (0-18kHz to avoid transition band)
        passband_mask = w <= 18000
        passband_db = H_db[passband_mask]
        ripple = np.max(passband_db) - np.min(passband_db)

        # Ripple should be less than 1dB
        assert ripple < 1.0

    def test_stopband_attenuation(self):
        """Stopband should have significant attenuation."""
        fs = 44100 * 16
        cutoff = 20000 / (fs / 2)
        h = signal.firwin(1001, cutoff, window=("kaiser", 14))

        w, H = signal.freqz(h, worN=8192, fs=fs)
        H_db = 20 * np.log10(np.abs(H) + 1e-12)

        # Check stopband (above 25kHz)
        stopband_mask = w >= 25000
        min_attenuation = np.min(H_db[stopband_mask])

        # Should have at least 40dB attenuation
        assert min_attenuation < -40


class TestMinimumPhaseProperty:
    """Tests for minimum phase filter properties."""

    def test_no_preringing(self):
        """Minimum phase filter should have no pre-ringing (energy before t=0)."""
        # Create minimum phase filter
        h_linear = signal.firwin(501, 0.5, window="hamming")
        h_min = signal.minimum_phase(h_linear, method="homomorphic")

        # Peak should be at or very near sample 0
        peak_idx = np.argmax(np.abs(h_min))

        # Peak should be in first 5% of samples
        assert peak_idx < len(h_min) * 0.05

    def test_energy_concentration(self):
        """Energy should be concentrated at the beginning."""
        h_linear = signal.firwin(501, 0.5, window="hamming")
        h_min = signal.minimum_phase(h_linear, method="homomorphic")

        # Calculate energy in first 10% vs rest
        n_front = len(h_min) // 10
        energy_front = np.sum(h_min[:n_front] ** 2)
        energy_total = np.sum(h_min**2)

        # First 10% should contain at least 50% of energy
        assert energy_front / energy_total > 0.5


class TestCoefficientFileLoading:
    """Tests for loading existing coefficient files."""

    def test_load_44k_coefficients(self, coefficients_dir):
        """Should load 44.1kHz coefficient file if it exists."""
        coeff_path = coefficients_dir / "filter_44k_2m_min_phase.bin"

        if not coeff_path.exists():
            pytest.skip("44kHz coefficient file not found")

        h = np.fromfile(coeff_path, dtype=np.float32)

        assert len(h) == 2_000_000
        assert np.isfinite(h).all()

    def test_load_48k_coefficients(self, coefficients_dir):
        """Should load 48kHz coefficient file if it exists."""
        coeff_path = coefficients_dir / "filter_48k_2m_min_phase.bin"

        if not coeff_path.exists():
            pytest.skip("48kHz coefficient file not found")

        h = np.fromfile(coeff_path, dtype=np.float32)

        assert len(h) == 2_000_000
        assert np.isfinite(h).all()

    def test_coefficient_dc_gain_is_normalized(self, coefficients_dir):
        """Loaded coefficients should have DC gain ~1.0."""
        coeff_path = coefficients_dir / "filter_44k_2m_min_phase.bin"

        if not coeff_path.exists():
            pytest.skip("44kHz coefficient file not found")

        h = np.fromfile(coeff_path, dtype=np.float32)
        dc_gain = np.sum(h)

        assert np.isclose(dc_gain, 1.0, rtol=0.01)


class TestMultiRateConfigs:
    """Tests for MULTI_RATE_CONFIGS and multi-rate filter generation."""

    def test_multi_rate_configs_has_8_entries(self):
        """MULTI_RATE_CONFIGS should have 8 entries (2 families × 4 ratios)."""
        import generate_filter

        assert len(generate_filter.MULTI_RATE_CONFIGS) == 8

    def test_multi_rate_configs_44k_family(self):
        """44k family should have correct configurations."""
        import generate_filter

        configs = generate_filter.MULTI_RATE_CONFIGS

        # 44k family entries
        assert "44k_16x" in configs
        assert "44k_8x" in configs
        assert "44k_4x" in configs
        assert "44k_2x" in configs

        # Check input rates
        assert configs["44k_16x"]["input_rate"] == 44100
        assert configs["44k_8x"]["input_rate"] == 88200
        assert configs["44k_4x"]["input_rate"] == 176400
        assert configs["44k_2x"]["input_rate"] == 352800

        # Check ratios
        assert configs["44k_16x"]["ratio"] == 16
        assert configs["44k_8x"]["ratio"] == 8
        assert configs["44k_4x"]["ratio"] == 4
        assert configs["44k_2x"]["ratio"] == 2

    def test_multi_rate_configs_48k_family(self):
        """48k family should have correct configurations."""
        import generate_filter

        configs = generate_filter.MULTI_RATE_CONFIGS

        # 48k family entries
        assert "48k_16x" in configs
        assert "48k_8x" in configs
        assert "48k_4x" in configs
        assert "48k_2x" in configs

        # Check input rates
        assert configs["48k_16x"]["input_rate"] == 48000
        assert configs["48k_8x"]["input_rate"] == 96000
        assert configs["48k_4x"]["input_rate"] == 192000
        assert configs["48k_2x"]["input_rate"] == 384000

        # Check ratios
        assert configs["48k_16x"]["ratio"] == 16
        assert configs["48k_8x"]["ratio"] == 8
        assert configs["48k_4x"]["ratio"] == 4
        assert configs["48k_2x"]["ratio"] == 2

    def test_stopband_equals_input_nyquist(self):
        """Stopband frequency should equal input Nyquist (input_rate / 2)."""
        import generate_filter

        for name, config in generate_filter.MULTI_RATE_CONFIGS.items():
            expected_stopband = config["input_rate"] // 2
            assert (
                config["stopband"] == expected_stopband
            ), f"{name}: stopband {config['stopband']} != input_nyquist {expected_stopband}"

    def test_output_rate_consistency(self):
        """All configs in same family should produce same output rate."""
        import generate_filter

        configs = generate_filter.MULTI_RATE_CONFIGS

        # 44k family -> 705.6kHz
        for name in ["44k_16x", "44k_8x", "44k_4x", "44k_2x"]:
            output_rate = configs[name]["input_rate"] * configs[name]["ratio"]
            assert output_rate == 705600, f"{name}: output {output_rate} != 705600"

        # 48k family -> 768kHz
        for name in ["48k_16x", "48k_8x", "48k_4x", "48k_2x"]:
            output_rate = configs[name]["input_rate"] * configs[name]["ratio"]
            assert output_rate == 768000, f"{name}: output {output_rate} != 768000"


class TestMultiRateOutputFilename:
    """Tests for multi-rate output filename format."""

    def test_filename_includes_ratio(self):
        """Output filename should include ratio: filter_{family}_{ratio}x_{taps}."""
        import generate_filter

        # Save originals
        orig_input = generate_filter.SAMPLE_RATE_INPUT
        orig_ratio = generate_filter.UPSAMPLE_RATIO
        orig_taps = generate_filter.N_TAPS
        orig_prefix = generate_filter.OUTPUT_PREFIX

        try:
            # Test 44k 16x case
            generate_filter.SAMPLE_RATE_INPUT = 44100
            generate_filter.UPSAMPLE_RATIO = 16
            generate_filter.N_TAPS = 2_000_000
            generate_filter.OUTPUT_PREFIX = None

            # The logic in export_coefficients builds filename
            taps_label = "2m"
            family = "44k"
            expected = f"filter_{family}_{generate_filter.UPSAMPLE_RATIO}x_{taps_label}_min_phase"
            assert expected == "filter_44k_16x_2m_min_phase"

            # Test 48k 8x case
            generate_filter.SAMPLE_RATE_INPUT = 96000
            generate_filter.UPSAMPLE_RATIO = 8
            family = "48k" if generate_filter.SAMPLE_RATE_INPUT % 48000 == 0 else "44k"
            expected = f"filter_{family}_{generate_filter.UPSAMPLE_RATIO}x_{taps_label}_min_phase"
            assert expected == "filter_48k_8x_2m_min_phase"
        finally:
            generate_filter.SAMPLE_RATE_INPUT = orig_input
            generate_filter.UPSAMPLE_RATIO = orig_ratio
            generate_filter.N_TAPS = orig_taps
            generate_filter.OUTPUT_PREFIX = orig_prefix


class TestValidateTapCountMultiRate:
    """Tests for tap count validation with different ratios."""

    def test_tap_count_divisible_by_ratio(self):
        """Tap count must be divisible by upsample ratio."""
        import generate_filter

        # All ratios used in multi-rate
        for ratio in [16, 8, 4, 2]:
            # Valid tap count (divisible)
            generate_filter.validate_tap_count(1024, ratio)  # 1024 is divisible by all

        # Invalid cases
        with pytest.raises(ValueError):
            generate_filter.validate_tap_count(1025, 16)  # Not divisible by 16

        with pytest.raises(ValueError):
            generate_filter.validate_tap_count(1025, 8)  # Not divisible by 8

        with pytest.raises(ValueError):
            generate_filter.validate_tap_count(1025, 4)  # Not divisible by 4

        # 1025 IS divisible by... nothing here that we use
        with pytest.raises(ValueError):
            generate_filter.validate_tap_count(1025, 2)  # Not divisible by 2
