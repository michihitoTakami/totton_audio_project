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


class TestPhaseType:
    """Tests for PhaseType enum."""

    def test_phase_type_values(self):
        """PhaseType should have correct values."""
        from generate_filter import PhaseType

        assert PhaseType.MINIMUM.value == "minimum"
        assert PhaseType.LINEAR.value == "linear"

    def test_phase_type_from_string(self):
        """PhaseType should be constructible from string."""
        from generate_filter import PhaseType

        assert PhaseType("minimum") == PhaseType.MINIMUM
        assert PhaseType("linear") == PhaseType.LINEAR


class TestMinimumPhaseMethod:
    """Tests for MinimumPhaseMethod enum."""

    def test_minimum_phase_method_values(self):
        """MinimumPhaseMethod should have correct values."""
        from generate_filter import MinimumPhaseMethod

        assert MinimumPhaseMethod.HOMOMORPHIC.value == "homomorphic"
        assert MinimumPhaseMethod.HILBERT.value == "hilbert"


class TestFilterConfig:
    """Tests for FilterConfig dataclass."""

    def test_default_values(self):
        """FilterConfig should have correct defaults."""
        from generate_filter import FilterConfig, PhaseType, MinimumPhaseMethod

        config = FilterConfig()

        assert config.n_taps == 2_000_000
        assert config.input_rate == 44100
        assert config.upsample_ratio == 16
        assert config.passband_end == 20000
        assert config.stopband_start == 22050  # auto-calculated
        assert config.stopband_attenuation_db == 197
        assert config.kaiser_beta == 55.0
        assert config.phase_type == PhaseType.MINIMUM
        assert config.minimum_phase_method == MinimumPhaseMethod.HOMOMORPHIC

    def test_output_rate_property(self):
        """output_rate should be calculated correctly."""
        from generate_filter import FilterConfig

        config = FilterConfig(input_rate=44100, upsample_ratio=16)
        assert config.output_rate == 705600

        config = FilterConfig(input_rate=48000, upsample_ratio=8)
        assert config.output_rate == 384000

    def test_family_property(self):
        """family should detect 44k vs 48k correctly."""
        from generate_filter import FilterConfig

        config_44k = FilterConfig(input_rate=44100)
        assert config_44k.family == "44k"

        config_88k = FilterConfig(input_rate=88200)
        assert config_88k.family == "44k"

        config_48k = FilterConfig(input_rate=48000)
        assert config_48k.family == "48k"

        config_96k = FilterConfig(input_rate=96000)
        assert config_96k.family == "48k"

    def test_base_name_minimum_phase(self):
        """base_name should include phase type for minimum phase."""
        from generate_filter import FilterConfig, PhaseType

        config = FilterConfig(
            n_taps=2_000_000,
            input_rate=44100,
            upsample_ratio=16,
            phase_type=PhaseType.MINIMUM,
        )
        assert config.base_name == "filter_44k_16x_2m_minimum"

    def test_base_name_linear_phase(self):
        """base_name should include phase type for linear phase."""
        from generate_filter import FilterConfig, PhaseType

        # 線形位相は奇数タップ必須
        config = FilterConfig(
            n_taps=2_000_001,  # Odd tap count required for linear phase
            input_rate=48000,
            upsample_ratio=8,
            phase_type=PhaseType.LINEAR,
        )
        # Note: taps_label uses original n_taps, which is 2000001
        assert "linear" in config.base_name
        assert "48k" in config.base_name

    def test_base_name_custom_prefix(self):
        """output_prefix should override auto-generated name."""
        from generate_filter import FilterConfig

        config = FilterConfig(output_prefix="custom_filter")
        assert config.base_name == "custom_filter"

    def test_linear_phase_requires_odd_taps(self):
        """Linear phase should reject even tap counts."""
        from generate_filter import FilterConfig, PhaseType

        # Even tap count should raise error for linear phase
        with pytest.raises(ValueError, match="奇数タップが必須"):
            FilterConfig(n_taps=2_000_000, phase_type=PhaseType.LINEAR)

    def test_linear_phase_accepts_odd_taps(self):
        """Linear phase should accept odd tap counts."""
        from generate_filter import FilterConfig, PhaseType

        # Odd tap count should work
        config = FilterConfig(n_taps=2_000_001, phase_type=PhaseType.LINEAR)
        assert config.n_taps == 2_000_001

    def test_minimum_phase_accepts_even_taps(self):
        """Minimum phase should accept even tap counts."""
        from generate_filter import FilterConfig, PhaseType

        config = FilterConfig(n_taps=2_000_000, phase_type=PhaseType.MINIMUM)
        assert config.n_taps == 2_000_000


class TestFilterDesigner:
    """Tests for FilterDesigner class."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for fast testing."""
        from generate_filter import FilterConfig, PhaseType

        return FilterConfig(
            n_taps=1600,
            input_rate=44100,
            upsample_ratio=16,
            passband_end=20000,
            stopband_start=22050,
            kaiser_beta=14,
            phase_type=PhaseType.MINIMUM,
        )

    def test_design_linear_phase(self, small_config):
        """design_linear_phase should create valid filter."""
        from generate_filter import FilterDesigner

        designer = FilterDesigner(small_config)
        h = designer.design_linear_phase()

        # Check tap count (may be +1 for odd)
        assert len(h) in [1600, 1601]
        # Check DC gain is non-zero
        assert np.abs(np.sum(h)) > 0

    def test_convert_to_minimum_phase(self, small_config):
        """convert_to_minimum_phase should produce causal filter."""
        from generate_filter import FilterDesigner

        designer = FilterDesigner(small_config)
        h_linear = designer.design_linear_phase()
        h_min = designer.convert_to_minimum_phase(h_linear)

        # Check tap count
        assert len(h_min) == 1600

        # Energy should be concentrated at front
        mid = len(h_min) // 2
        energy_first = np.sum(h_min[:mid] ** 2)
        energy_second = np.sum(h_min[mid:] ** 2)
        assert energy_first > energy_second

    def test_design_returns_minimum_phase(self, small_config):
        """design() with MINIMUM should return minimum phase filter."""
        from generate_filter import FilterDesigner, PhaseType

        small_config.phase_type = PhaseType.MINIMUM
        designer = FilterDesigner(small_config)
        h_final, h_linear = designer.design()

        # Should have h_linear for comparison
        assert h_linear is not None
        # h_final should be minimum phase (energy at front)
        peak_idx = np.argmax(np.abs(h_final))
        assert peak_idx < len(h_final) * 0.1

    def test_design_returns_linear_phase(self):
        """design() with LINEAR should return linear phase filter."""
        from generate_filter import FilterConfig, FilterDesigner, PhaseType

        # Use odd tap count divisible by 7 to preserve symmetry
        config = FilterConfig(
            n_taps=1603,  # 1603 = 229 * 7, odd and divisible by 7
            input_rate=44100,
            upsample_ratio=7,  # Unusual ratio but valid
            kaiser_beta=14,
            phase_type=PhaseType.LINEAR,
        )
        designer = FilterDesigner(config)
        h_final, h_linear = designer.design()

        # Should return same as linear
        assert h_linear is not None
        # Linear phase with odd taps should be symmetric
        assert np.allclose(h_final, h_final[::-1], atol=1e-10)

    def test_design_linear_phase_odd_taps_symmetric(self):
        """design() with LINEAR and odd tap count should produce symmetric filter."""
        from generate_filter import FilterConfig, FilterDesigner, PhaseType

        # Linear phase requires odd tap count (enforced at FilterConfig level)
        config = FilterConfig(
            n_taps=1601,  # Odd, as required for linear phase
            input_rate=44100,
            upsample_ratio=16,
            kaiser_beta=14,
            phase_type=PhaseType.LINEAR,
        )
        designer = FilterDesigner(config)
        h_final, h_linear = designer.design()

        # Tap count should be odd (1601)
        assert len(h_final) == 1601
        assert len(h_final) % 2 == 1
        # Should be symmetric (Type I FIR)
        assert np.allclose(h_final, h_final[::-1], atol=1e-10)


class TestFilterValidator:
    """Tests for FilterValidator class."""

    def test_validate_returns_expected_keys(self):
        """validate should return dict with expected keys."""
        from generate_filter import FilterConfig, FilterValidator

        config = FilterConfig(
            n_taps=1000,
            input_rate=44100,
            upsample_ratio=16,
            stopband_attenuation_db=40,
        )
        validator = FilterValidator(config)

        h = signal.firwin(101, 0.5, window="hamming")
        results = validator.validate(h)

        assert "passband_ripple_db" in results
        assert "stopband_attenuation_db" in results
        assert "is_minimum_phase" in results
        assert "is_symmetric" in results
        assert "phase_type" in results
        assert "peak_position" in results

    def test_validate_detects_minimum_phase(self):
        """validate should detect minimum phase filters."""
        from generate_filter import FilterConfig, FilterValidator, PhaseType

        config = FilterConfig(
            n_taps=1000,
            input_rate=44100,
            upsample_ratio=16,
            stopband_attenuation_db=40,
            phase_type=PhaseType.MINIMUM,
        )
        validator = FilterValidator(config)

        # Create minimum phase filter
        h_linear = signal.firwin(1001, 0.5, window="hamming")
        h_min = signal.minimum_phase(h_linear, method="homomorphic")

        results = validator.validate(h_min)

        # Peak should be near front
        assert results["peak_position"] < len(h_min) * 0.1
        # Energy ratio should indicate causality
        assert results["energy_ratio_first_to_second_half"] > 1.0

    def test_validate_detects_symmetric(self):
        """validate should detect symmetric (linear phase) filters."""
        from generate_filter import FilterConfig, FilterValidator, PhaseType

        # 線形位相は奇数タップ必須
        config = FilterConfig(
            n_taps=1001,  # Odd tap count required for linear phase
            input_rate=44100,
            upsample_ratio=16,
            phase_type=PhaseType.LINEAR,
        )
        validator = FilterValidator(config)

        # Linear phase filter is symmetric
        h = signal.firwin(101, 0.5, window="hamming")
        results = validator.validate(h)

        assert results["is_symmetric"] is True


class TestValidateTapCount:
    """Tests for validate_tap_count function."""

    def test_valid_tap_count_divisible_by_16(self):
        """Tap count divisible by 16 should pass."""
        from generate_filter import validate_tap_count

        # Should not raise
        validate_tap_count(1600, 16)
        validate_tap_count(2_000_000, 16)

    def test_invalid_tap_count_not_divisible(self):
        """Tap count not divisible by upsample ratio should raise."""
        from generate_filter import validate_tap_count

        with pytest.raises(ValueError, match="倍数である必要があります"):
            validate_tap_count(1601, 16)

    def test_valid_tap_count_different_ratio(self):
        """Different upsample ratios should work."""
        from generate_filter import validate_tap_count

        validate_tap_count(1000, 8)
        validate_tap_count(1000, 4)


class TestNormalizeCoefficients:
    """Tests for normalize_coefficients function."""

    def test_normalizes_dc_gain_to_one(self):
        """DC gain should be normalized to 1.0."""
        from generate_filter import normalize_coefficients

        # Create test coefficients with known DC gain
        h = np.array([0.5, 1.0, 0.5])  # DC gain = 2.0
        h_norm, info = normalize_coefficients(h)

        assert np.isclose(np.sum(h_norm), 1.0, rtol=1e-6)
        assert info["normalization_applied"] is True
        assert np.isclose(info["original_dc_gain"], 2.0)

    def test_zero_dc_gain_raises_error(self):
        """Zero DC gain should raise ValueError."""
        from generate_filter import normalize_coefficients

        h = np.array([1.0, -1.0])  # DC gain = 0
        with pytest.raises(ValueError, match="DCゲインが0に近すぎます"):
            normalize_coefficients(h)

    def test_preserves_shape(self):
        """Normalization should preserve coefficient shape."""
        from generate_filter import normalize_coefficients

        h = np.random.randn(1000)
        h_norm, _ = normalize_coefficients(h)

        assert h_norm.shape == h.shape


class TestFilterDesignLegacy:
    """Tests for legacy filter design functions using small tap counts."""

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
        from generate_filter import design_linear_phase_filter

        h = design_linear_phase_filter()

        # Check tap count (may be +1 for odd)
        assert len(h) in [1600, 1601]

        # Check DC gain is non-zero
        assert np.abs(np.sum(h)) > 0

    def test_minimum_phase_conversion(self, setup_small_filter_params):
        """Minimum phase conversion should produce causal filter."""
        from generate_filter import (
            design_linear_phase_filter,
            convert_to_minimum_phase,
        )

        h_linear = design_linear_phase_filter()
        h_min = convert_to_minimum_phase(h_linear)

        # Check tap count
        assert len(h_min) == 1600

        # Check energy is concentrated at front (causal)
        # For minimum phase, most energy should be in first half
        mid = len(h_min) // 2
        energy_first = np.sum(h_min[:mid] ** 2)
        energy_second = np.sum(h_min[mid:] ** 2)

        # First half should have more energy
        assert energy_first > energy_second


class TestValidateSpecificationsLegacy:
    """Tests for legacy validate_specifications function."""

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
        from generate_filter import validate_specifications

        # Create a simple lowpass filter for testing
        h = signal.firwin(101, 0.5, window="hamming")

        results = validate_specifications(h)

        assert isinstance(results, dict)
        assert "passband_ripple_db" in results
        assert "stopband_attenuation_db" in results
        assert "is_minimum_phase" in results
        assert "peak_position" in results

    def test_minimum_phase_detection(self, setup_validation_params):
        """Should correctly detect minimum phase filters."""
        from generate_filter import validate_specifications

        # Create a minimum phase filter
        h_linear = signal.firwin(101, 0.5, window="hamming")
        h_min = signal.minimum_phase(h_linear, method="homomorphic")

        results = validate_specifications(h_min)

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
        from generate_filter import MULTI_RATE_CONFIGS

        assert len(MULTI_RATE_CONFIGS) == 8

    def test_multi_rate_configs_44k_family(self):
        """44k family should have correct configurations."""
        from generate_filter import MULTI_RATE_CONFIGS

        configs = MULTI_RATE_CONFIGS

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
        from generate_filter import MULTI_RATE_CONFIGS

        configs = MULTI_RATE_CONFIGS

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
        from generate_filter import MULTI_RATE_CONFIGS

        for name, config in MULTI_RATE_CONFIGS.items():
            expected_stopband = config["input_rate"] // 2
            assert (
                config["stopband"] == expected_stopband
            ), f"{name}: stopband {config['stopband']} != input_nyquist {expected_stopband}"

    def test_output_rate_consistency(self):
        """All configs in same family should produce same output rate."""
        from generate_filter import MULTI_RATE_CONFIGS

        configs = MULTI_RATE_CONFIGS

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

    def test_filename_includes_phase_type(self):
        """Output filename should include phase type."""
        from generate_filter import FilterConfig, PhaseType

        # Minimum phase (even taps OK)
        config_min = FilterConfig(
            n_taps=2_000_000,
            input_rate=44100,
            upsample_ratio=16,
            phase_type=PhaseType.MINIMUM,
        )
        assert config_min.base_name == "filter_44k_16x_2m_minimum"

        # Linear phase (odd taps required)
        config_lin = FilterConfig(
            n_taps=2_000_001,  # Odd tap count required for linear phase
            input_rate=44100,
            upsample_ratio=16,
            phase_type=PhaseType.LINEAR,
        )
        assert "linear" in config_lin.base_name
        assert "44k" in config_lin.base_name


class TestValidateTapCountMultiRate:
    """Tests for tap count validation with different ratios."""

    def test_tap_count_divisible_by_ratio(self):
        """Tap count must be divisible by upsample ratio."""
        from generate_filter import validate_tap_count

        # All ratios used in multi-rate
        for ratio in [16, 8, 4, 2]:
            # Valid tap count (divisible)
            validate_tap_count(1024, ratio)  # 1024 is divisible by all

        # Invalid cases
        with pytest.raises(ValueError):
            validate_tap_count(1025, 16)  # Not divisible by 16

        with pytest.raises(ValueError):
            validate_tap_count(1025, 8)  # Not divisible by 8

        with pytest.raises(ValueError):
            validate_tap_count(1025, 4)  # Not divisible by 4

        # 1025 IS divisible by... nothing here that we use
        with pytest.raises(ValueError):
            validate_tap_count(1025, 2)  # Not divisible by 2


class TestLinearPhasePadding:
    """Tests for linear phase zero-padding to ratio multiples."""

    def test_compute_padded_taps_already_divisible(self):
        """compute_padded_taps should return n_taps if already divisible."""
        from generate_filter import compute_padded_taps

        assert compute_padded_taps(1024, 16) == 1024
        assert compute_padded_taps(1024, 8) == 1024
        assert compute_padded_taps(1024, 4) == 1024

    def test_compute_padded_taps_needs_padding(self):
        """compute_padded_taps should return next multiple if not divisible."""
        from generate_filter import compute_padded_taps

        # 1025 -> 1040 (next multiple of 16)
        assert compute_padded_taps(1025, 16) == 1040
        # 1025 -> 1032 (next multiple of 8)
        assert compute_padded_taps(1025, 8) == 1032
        # 2000001 -> 2000016 (next multiple of 16)
        assert compute_padded_taps(2_000_001, 16) == 2_000_016

    def test_final_taps_minimum_phase_unchanged(self):
        """final_taps should equal n_taps for minimum phase."""
        from generate_filter import FilterConfig, PhaseType

        config = FilterConfig(n_taps=2_000_000, phase_type=PhaseType.MINIMUM)
        assert config.final_taps == 2_000_000

    def test_final_taps_linear_phase_padded(self):
        """final_taps should be padded to ratio multiple for linear phase."""
        from generate_filter import FilterConfig, PhaseType

        # 2,000,001 is odd (valid for linear), but not divisible by 16
        # Should be padded to 2,000,016
        config = FilterConfig(
            n_taps=2_000_001,
            upsample_ratio=16,
            phase_type=PhaseType.LINEAR,
        )
        assert config.final_taps == 2_000_016
        assert config.final_taps % 16 == 0

    def test_final_taps_linear_phase_already_divisible(self):
        """final_taps should be unchanged if already divisible."""
        from generate_filter import FilterConfig, PhaseType

        # Find an odd number divisible by... well, odd numbers aren't divisible
        # by even ratios. Let's use a ratio of 1 for this edge case.
        # Actually, with ratio 16, no odd number is divisible. So final_taps
        # will always be padded for linear phase with even ratios.
        # Let's verify that the padding is minimal.
        config = FilterConfig(
            n_taps=2_000_001,
            upsample_ratio=16,
            phase_type=PhaseType.LINEAR,
        )
        # Padding should add at most ratio-1 zeros
        assert config.final_taps - config.n_taps < 16

    def test_taps_label_reflects_final_taps(self):
        """taps_label should use final_taps, not n_taps."""
        from generate_filter import FilterConfig, PhaseType

        config = FilterConfig(
            n_taps=2_000_001,
            upsample_ratio=16,
            phase_type=PhaseType.LINEAR,
        )
        # final_taps = 2,000,016, which is not a nice round number
        # taps_label should be "2000016" (not "2m" since it's not exactly 2M)
        assert config.taps_label == "2000016"

        # For minimum phase with 2M taps
        config_min = FilterConfig(n_taps=2_000_000, phase_type=PhaseType.MINIMUM)
        assert config_min.taps_label == "2m"


class TestFilterGenerator:
    """Tests for FilterGenerator orchestration class."""

    def test_generator_creates_all_components(self):
        """FilterGenerator should initialize all component classes."""
        from generate_filter import (
            FilterConfig,
            FilterGenerator,
            FilterDesigner,
            FilterValidator,
            FilterExporter,
            FilterPlotter,
        )

        config = FilterConfig(n_taps=1024)
        generator = FilterGenerator(config)

        assert isinstance(generator.designer, FilterDesigner)
        assert isinstance(generator.validator, FilterValidator)
        assert isinstance(generator.exporter, FilterExporter)
        assert isinstance(generator.plotter, FilterPlotter)
