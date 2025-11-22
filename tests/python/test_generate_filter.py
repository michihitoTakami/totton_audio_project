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
