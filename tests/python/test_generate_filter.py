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
        assert PhaseType.HYBRID.value == "hybrid"

    def test_phase_type_from_string(self):
        """PhaseType should be constructible from string."""
        from generate_filter import PhaseType

        assert PhaseType("minimum") == PhaseType.MINIMUM
        assert PhaseType("hybrid") == PhaseType.HYBRID


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

        assert config.n_taps == 640_000
        assert config.input_rate == 44100
        assert config.upsample_ratio == 16
        assert config.passband_end == 20000
        assert config.stopband_start == 22050  # auto-calculated
        assert (
            config.stopband_attenuation_db == 160
        )  # Updated: realistic target for min phase
        assert config.kaiser_beta == 28.0
        assert config.phase_type == PhaseType.MINIMUM
        assert config.minimum_phase_method == MinimumPhaseMethod.HOMOMORPHIC
        assert config.target_dc_gain == config.upsample_ratio
        # dc_gain_factor: 全レートで音量統一（デフォルト0.99）
        assert config.dc_gain_factor == 0.99

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
        # C++ expects: filter_{family}_{ratio}x_{taps}_min_phase.bin
        assert config.base_name == "filter_44k_16x_2m_min_phase"

    def test_base_name_hybrid_phase(self):
        """base_name should include phase type label for hybrid phase."""
        from generate_filter import FilterConfig, PhaseType

        config = FilterConfig(
            n_taps=2_000_000,
            input_rate=48000,
            upsample_ratio=8,
            phase_type=PhaseType.HYBRID,
        )
        assert config.phase_label == "hybrid_phase"
        assert config.base_name.endswith("_hybrid_phase")

    def test_base_name_custom_prefix(self):
        """output_prefix should override auto-generated name."""
        from generate_filter import FilterConfig

        config = FilterConfig(output_prefix="custom_filter")
        assert config.base_name == "custom_filter"

    def test_hybrid_delay_samples_property(self):
        """Hybrid delay samples should match delay ms."""
        from generate_filter import FilterConfig, PhaseType

        config = FilterConfig(
            n_taps=8192,
            input_rate=44100,
            upsample_ratio=16,
            phase_type=PhaseType.HYBRID,
            hybrid_delay_ms=5.0,
        )
        expected = int(round(0.005 * 44100 * 16))
        assert config.hybrid_delay_samples == expected

    def test_hybrid_delay_validation(self):
        """Hybrid delay must be shorter than tap length."""
        from generate_filter import FilterConfig, PhaseType
        import pytest

        with pytest.raises(ValueError):
            FilterConfig(
                n_taps=2048,
                input_rate=44100,
                upsample_ratio=16,
                phase_type=PhaseType.HYBRID,
            )

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

    @pytest.fixture
    def hybrid_config(self):
        """Hybrid config with enough taps for 10ms delay."""
        from generate_filter import FilterConfig, PhaseType

        return FilterConfig(
            n_taps=8192,
            input_rate=44100,
            upsample_ratio=16,
            passband_end=20000,
            stopband_start=22050,
            kaiser_beta=14,
            phase_type=PhaseType.HYBRID,
            hybrid_fast_window_samples=4096,
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

    def test_design_returns_hybrid_phase(self, hybrid_config):
        """design() with HYBRID should align peak near delay samples."""
        from generate_filter import FilterDesigner

        designer = FilterDesigner(hybrid_config)
        h_final, h_linear = designer.design()

        assert h_linear is not None
        assert len(h_final) == hybrid_config.n_taps

        peak_idx = int(np.argmax(np.abs(h_final)))
        delta = abs(peak_idx - hybrid_config.hybrid_delay_samples)
        # Allow 2 ms tolerance due to windowing
        tolerance = int(hybrid_config.output_rate * 0.002)
        assert delta < tolerance

    def test_hybrid_gain_alignment_metrics(self, hybrid_config):
        """Hybrid設計時に正規化ゲインと群遅延誤差が記録される"""
        from generate_filter import FilterDesigner

        designer = FilterDesigner(hybrid_config)
        designer.design()

        info = designer.hybrid_gain_alignment
        assert info is not None
        assert info["normalization_gain"] > 0
        assert np.isclose(
            info["passband_reference"],
            info["passband_hybrid"],
            rtol=0.1,
        )
        if info["allpass_solver_status"] == "blend_fallback":
            assert info["group_delay_error_ms"] < 7.0
        else:
            assert info["group_delay_error_ms"] < 5.0
            assert info["allpass_sections"] == hybrid_config.hybrid_allpass_sections
            assert info["allpass_solver_status"] == "success"
            assert info["allpass_rmse_ms"] < 2.0

    def test_hybrid_group_delay_targets(self, hybrid_config):
        """Hybrid設計時に高域群遅延がターゲット値へ収束する"""
        from generate_filter import FilterDesigner

        designer = FilterDesigner(hybrid_config)
        h_final, _ = designer.design()

        n_fft = 2 ** int(np.ceil(np.log2(len(h_final) * 2)))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / hybrid_config.output_rate)
        omega = 2 * np.pi * freqs
        H = np.fft.rfft(h_final, n=n_fft)
        phase = np.unwrap(np.angle(H))
        tau = designer._compute_group_delay(phase, omega)

        transition_edge = (
            hybrid_config.hybrid_crossover_hz + hybrid_config.hybrid_transition_hz
        )
        high_mask = freqs >= transition_edge
        assert np.any(high_mask)

        tau_high = tau[high_mask]
        target_delay = hybrid_config.hybrid_delay_ms / 1000.0
        mean_error = float(np.mean(np.abs(tau_high - target_delay)))
        info = designer.hybrid_gain_alignment or {}
        status = info.get("allpass_solver_status", "success")
        limit_ms = 2.0 if status != "blend_fallback" else 7.0
        assert mean_error * 1000.0 < limit_ms


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

        config = FilterConfig(
            n_taps=1001,  # Odd tap count required for linear phase
            input_rate=44100,
            upsample_ratio=16,
            phase_type=PhaseType.MINIMUM,
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
    """Tests for normalize_coefficients function.

    #259追加修正: DCゲイン統一（L × dc_gain_factor）+ L1ノルム出力。
    全レートで音量を統一し、L1ノルムはグローバル安全ゲイン計算用に出力。
    """

    def test_normalizes_dc_gain_with_factor(self):
        """DCゲインが target × dc_gain_factor に正規化されること"""
        from generate_filter import normalize_coefficients

        # Create test coefficients with known DC gain
        h = np.array([0.5, 1.0, 0.5])  # DC gain = 2.0
        h_norm, info = normalize_coefficients(
            h, target_dc_gain=1.0, dc_gain_factor=0.99
        )

        # DCゲインが目標値（1.0 × 0.99 = 0.99）に正規化される
        assert np.isclose(np.sum(h_norm), 0.99, rtol=1e-6)
        assert info["normalization_applied"] is True
        assert np.isclose(info["original_dc_gain"], 2.0)
        assert np.isclose(info["target_dc_gain"], 1.0)
        assert np.isclose(info["dc_gain_factor"], 0.99)
        assert np.isclose(info["normalized_dc_gain"], 0.99, rtol=1e-6)

    def test_l1_norm_output(self):
        """L1ノルムがメタデータに出力されること"""
        from generate_filter import normalize_coefficients

        h = np.array([0.5, -0.3, 0.2])  # DC gain = 0.4, L1 = 1.0
        h_norm, info = normalize_coefficients(h, target_dc_gain=1.0, dc_gain_factor=1.0)

        # L1ノルムが出力される
        assert "l1_norm" in info
        assert "l1_norm_ratio" in info
        # L1 = sum(|h_norm|)
        expected_l1 = np.sum(np.abs(h_norm))
        assert np.isclose(info["l1_norm"], expected_l1, rtol=1e-6)
        # L1/L ratio
        assert np.isclose(info["l1_norm_ratio"], expected_l1 / 1.0, rtol=1e-6)

    def test_normalizes_to_upsample_ratio(self):
        """アップサンプル比に応じたDCゲイン正規化（dc_gain_factor適用）"""
        from generate_filter import normalize_coefficients

        h = np.array([0.1, 0.1])  # DC gain = 0.2
        target = 16.0  # 16x upsample
        h_norm, info = normalize_coefficients(
            h, target_dc_gain=target, dc_gain_factor=0.99
        )

        # DCゲインが16.0 × 0.99 = 15.84に正規化される
        expected_dc = 16.0 * 0.99
        assert np.isclose(np.sum(h_norm), expected_dc, rtol=1e-6)
        assert np.isclose(info["normalized_dc_gain"], expected_dc, rtol=1e-6)
        # スケール = 15.84 / 0.2 = 79.2
        assert np.isclose(info["applied_scale"], expected_dc / 0.2, rtol=1e-6)

    def test_dc_gain_factor_1_equals_full_target(self):
        """dc_gain_factor=1.0でフル目標値になること"""
        from generate_filter import normalize_coefficients

        h = np.array([1.0, 1.0])  # DC gain = 2.0
        target = 4.0
        h_norm, info = normalize_coefficients(
            h, target_dc_gain=target, dc_gain_factor=1.0
        )

        # DCゲインは正確に4.0
        assert np.isclose(np.sum(h_norm), 4.0, rtol=1e-6)
        assert np.isclose(info["normalized_dc_gain"], 4.0, rtol=1e-6)

    def test_invalid_dc_gain_factor_raises_error(self):
        """不正なdc_gain_factorでエラー"""
        from generate_filter import normalize_coefficients

        h = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="dc_gain_factor"):
            normalize_coefficients(h, dc_gain_factor=0.0)
        with pytest.raises(ValueError, match="dc_gain_factor"):
            normalize_coefficients(h, dc_gain_factor=1.5)

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

    def test_invalid_target_raises_error(self):
        """Non-positive target DC gain should raise ValueError."""
        from generate_filter import normalize_coefficients

        with pytest.raises(ValueError, match="DCゲインのターゲット"):
            normalize_coefficients(np.array([1.0]), target_dc_gain=0.0)


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

    def test_load_44k_16x_coefficients(self, coefficients_dir):
        """Should load 44.1kHz 16x coefficient file if it exists."""
        coeff_path = coefficients_dir / "filter_44k_16x_2m_hybrid_phase.bin"

        if not coeff_path.exists():
            pytest.skip("44kHz 16x coefficient file not found")

        h = np.fromfile(coeff_path, dtype=np.float32)

        assert len(h) == 640_000
        assert np.isfinite(h).all()

    def test_load_48k_16x_coefficients(self, coefficients_dir):
        """Should load 48kHz 16x coefficient file if it exists."""
        coeff_path = coefficients_dir / "filter_48k_16x_2m_hybrid_phase.bin"

        if not coeff_path.exists():
            pytest.skip("48kHz 16x coefficient file not found")

        h = np.fromfile(coeff_path, dtype=np.float32)

        assert len(h) == 640_000
        assert np.isfinite(h).all()

    def test_coefficient_dc_gain_matches_ratio(self, coefficients_dir):
        """Loaded coefficients should have DC gain close to target (L * 0.99)."""
        coeff_path = coefficients_dir / "filter_44k_16x_2m_hybrid_phase.bin"

        if not coeff_path.exists():
            pytest.skip("44kHz 16x coefficient file not found")

        h = np.fromfile(coeff_path, dtype=np.float32)
        dc_gain = np.sum(h)

        # DCゲイン = 16 * 0.99 = 15.84
        assert np.isclose(dc_gain, 15.84, rtol=0.01)


class TestCoefficientFileNaming:
    """Tests for coefficient file naming convention (2m format)."""

    # Expected filenames for all 8 multi-rate configurations
    EXPECTED_FILENAMES = [
        "filter_44k_16x_2m_hybrid_phase.bin",
        "filter_44k_8x_2m_hybrid_phase.bin",
        "filter_44k_4x_2m_hybrid_phase.bin",
        "filter_44k_2x_2m_hybrid_phase.bin",
        "filter_48k_16x_2m_hybrid_phase.bin",
        "filter_48k_8x_2m_hybrid_phase.bin",
        "filter_48k_4x_2m_hybrid_phase.bin",
        "filter_48k_2x_2m_hybrid_phase.bin",
    ]

    def test_all_coefficient_files_exist(self, coefficients_dir):
        """All 8 coefficient files should exist with 2m naming convention."""
        missing_files = []
        for filename in self.EXPECTED_FILENAMES:
            if not (coefficients_dir / filename).exists():
                missing_files.append(filename)

        assert not missing_files, f"Missing coefficient files: {missing_files}"

    def test_coefficient_filenames_use_2m_format(self, coefficients_dir):
        """Coefficient files should use '2m' instead of '2000000' in filenames."""
        # Check that old naming convention files don't exist
        old_format_files = list(coefficients_dir.glob("*_2000000_*.bin"))
        assert not old_format_files, f"Found files with old '2000000' naming: {[f.name for f in old_format_files]}"

        # Check that new naming convention files exist
        new_format_files = list(coefficients_dir.glob("*_2m_*.bin"))
        assert (
            len(new_format_files) >= 8
        ), f"Expected at least 8 files with '2m' naming, found {len(new_format_files)}"

    def test_json_metadata_files_match_bin_files(self, coefficients_dir):
        """Each .bin file should have a corresponding .json metadata file."""
        for filename in self.EXPECTED_FILENAMES:
            bin_path = coefficients_dir / filename
            json_path = coefficients_dir / filename.replace(".bin", ".json")

            if bin_path.exists():
                assert json_path.exists(), f"Missing JSON metadata for {filename}"

    def test_filter_config_generates_2m_filename(self):
        """FilterConfig should generate filenames with '2m' for 2M taps."""
        from generate_filter import FilterConfig, PhaseType

        # Test all 8 configurations
        test_cases = [
            (44100, 16, "filter_44k_16x_2m_min_phase"),
            (88200, 8, "filter_44k_8x_2m_min_phase"),
            (176400, 4, "filter_44k_4x_2m_min_phase"),
            (352800, 2, "filter_44k_2x_2m_min_phase"),
            (48000, 16, "filter_48k_16x_2m_min_phase"),
            (96000, 8, "filter_48k_8x_2m_min_phase"),
            (192000, 4, "filter_48k_4x_2m_min_phase"),
            (384000, 2, "filter_48k_2x_2m_min_phase"),
        ]

        for input_rate, ratio, expected_basename in test_cases:
            config = FilterConfig(
                n_taps=2_000_000,
                input_rate=input_rate,
                upsample_ratio=ratio,
                phase_type=PhaseType.MINIMUM,
            )
            assert (
                config.base_name == expected_basename
            ), f"Expected {expected_basename}, got {config.base_name}"

    def test_non_2m_taps_use_numeric_format(self):
        """Non-2M tap counts should use numeric format in filename."""
        from generate_filter import FilterConfig, PhaseType

        # Test with different tap counts
        config_1m = FilterConfig(
            n_taps=1_000_000,
            input_rate=44100,
            upsample_ratio=16,
            phase_type=PhaseType.MINIMUM,
        )
        assert "1000000" in config_1m.base_name
        assert "1m" not in config_1m.base_name

        config_500k = FilterConfig(
            n_taps=500_000,
            input_rate=44100,
            upsample_ratio=16,
            phase_type=PhaseType.MINIMUM,
        )
        assert "500000" in config_500k.base_name


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
        # C++ expects: filter_{family}_{ratio}x_{taps}_min_phase.bin
        assert config_min.base_name == "filter_44k_16x_2m_min_phase"

        # Hybrid phase (same taps as minimum)
        config_hybrid = FilterConfig(
            n_taps=2_000_000,
            input_rate=44100,
            upsample_ratio=16,
            phase_type=PhaseType.HYBRID,
        )
        assert config_hybrid.base_name.endswith("_hybrid_phase")
        assert "44k" in config_hybrid.base_name


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


class TestPhaseTypeErrorHandling:
    """Tests for error handling with invalid phase type parameters."""

    def test_invalid_phase_type_string_raises_error(self):
        """PhaseType should raise ValueError for invalid string."""
        from generate_filter import PhaseType

        with pytest.raises(ValueError):
            PhaseType("invalid")

        with pytest.raises(ValueError):
            PhaseType("mixed")

        with pytest.raises(ValueError):
            PhaseType("")

    def test_invalid_minimum_phase_method_string_raises_error(self):
        """MinimumPhaseMethod should raise ValueError for invalid string."""
        from generate_filter import MinimumPhaseMethod

        with pytest.raises(ValueError):
            MinimumPhaseMethod("invalid")

        with pytest.raises(ValueError):
            MinimumPhaseMethod("")


class TestFilterConfigErrorHandling:
    """Tests for error handling with invalid FilterConfig parameters."""

    def test_zero_tap_count_raises_error(self):
        """FilterConfig should raise ValueError for zero tap count."""
        from generate_filter import FilterConfig

        with pytest.raises(ValueError):
            FilterConfig(n_taps=0)

    def test_negative_tap_count_raises_error(self):
        """FilterConfig should raise ValueError for negative tap count."""
        from generate_filter import FilterConfig

        with pytest.raises(ValueError):
            FilterConfig(n_taps=-1000)

    def test_zero_input_rate_raises_error(self):
        """FilterConfig should raise ValueError for zero input rate."""
        from generate_filter import FilterConfig

        with pytest.raises(ValueError):
            FilterConfig(input_rate=0)

    def test_negative_input_rate_raises_error(self):
        """FilterConfig should raise ValueError for negative input rate."""
        from generate_filter import FilterConfig

        with pytest.raises(ValueError):
            FilterConfig(input_rate=-44100)

    def test_zero_upsample_ratio_raises_error(self):
        """FilterConfig should raise ValueError for zero upsample ratio."""
        from generate_filter import FilterConfig

        with pytest.raises(ValueError):
            FilterConfig(upsample_ratio=0)

    def test_negative_upsample_ratio_raises_error(self):
        """FilterConfig should raise ValueError for negative upsample ratio."""
        from generate_filter import FilterConfig

        with pytest.raises(ValueError):
            FilterConfig(upsample_ratio=-16)

    def test_negative_kaiser_beta_raises_error(self):
        """FilterConfig should raise ValueError for negative kaiser beta."""
        from generate_filter import FilterConfig

        with pytest.raises(ValueError):
            FilterConfig(kaiser_beta=-1.0)

    def test_passband_exceeds_nyquist_raises_error(self):
        """FilterConfig should raise ValueError if passband > Nyquist."""
        from generate_filter import FilterConfig

        # Passband (25kHz) > Nyquist (22.05kHz) for 44.1kHz input
        with pytest.raises(ValueError):
            FilterConfig(input_rate=44100, passband_end=25000)

    def test_stopband_less_than_passband_raises_error(self):
        """FilterConfig should raise ValueError if stopband < passband."""
        from generate_filter import FilterConfig

        with pytest.raises(ValueError):
            FilterConfig(passband_end=20000, stopband_start=18000)

    def test_stopband_exceeds_output_nyquist_raises_error(self):
        """FilterConfig should raise ValueError if stopband >= output Nyquist."""
        from generate_filter import FilterConfig

        # output_nyquist = 44100 * 16 / 2 = 352800 Hz
        # stopband_start = 1000000 Hz > 352800 Hz -> エラー
        with pytest.raises(ValueError, match="出力ナイキスト周波数"):
            FilterConfig(
                input_rate=44100,
                upsample_ratio=16,
                stopband_start=1_000_000,
            )

    def test_stopband_at_output_nyquist_raises_error(self):
        """FilterConfig should raise ValueError if stopband == output Nyquist."""
        from generate_filter import FilterConfig

        # output_nyquist = 44100 * 16 / 2 = 352800 Hz
        with pytest.raises(ValueError, match="出力ナイキスト周波数"):
            FilterConfig(
                input_rate=44100,
                upsample_ratio=16,
                stopband_start=352800,
            )


class TestNormalizeCoefficientsErrorHandling:
    """Tests for error handling in normalize_coefficients."""

    def test_empty_array_raises_error(self):
        """normalize_coefficients should raise ValueError for empty array."""
        from generate_filter import normalize_coefficients

        with pytest.raises(ValueError):
            normalize_coefficients(np.array([]))

    def test_near_zero_dc_gain_raises_error(self):
        """normalize_coefficients should raise ValueError for near-zero DC gain."""
        from generate_filter import normalize_coefficients

        # Array with DC gain very close to zero
        h = np.array([1.0, -1.0, 1e-15])
        with pytest.raises(ValueError, match="DCゲインが0に近すぎます"):
            normalize_coefficients(h)


class TestCoefficientDcGain:
    """Tests for shipped coefficient DC gain after upsample normalization.

    新形式のフィルタは DCゲイン = L × 0.99 で統一されている。
    """

    def test_coefficient_dc_matches_target(self):
        """Shipped filters should have DC gain = L * 0.99."""
        coeff_dir = Path(__file__).parent.parent.parent / "data" / "coefficients"
        # 新形式フィルタ: DCゲイン = L * 0.99
        cases = [
            ("filter_44k_16x_2m_hybrid_phase.bin", 16.0 * 0.99),  # 15.84
            ("filter_48k_16x_2m_hybrid_phase.bin", 16.0 * 0.99),  # 15.84
            ("filter_44k_8x_2m_hybrid_phase.bin", 8.0 * 0.99),  # 7.92
            ("filter_48k_8x_2m_hybrid_phase.bin", 8.0 * 0.99),  # 7.92
        ]
        for filename, expected_dc in cases:
            filepath = coeff_dir / filename
            if not filepath.exists():
                pytest.skip(f"{filename} not found")
            data = np.fromfile(filepath, dtype=np.float32)
            dc_gain = float(np.sum(data))
            assert np.isclose(
                dc_gain, expected_dc, rtol=1e-3
            ), f"{filename}: expected DC={expected_dc:.4f}, got {dc_gain:.4f}"


class TestValidateTapCountErrorHandling:
    """Tests for error handling in validate_tap_count."""

    def test_non_multiple_ratio_raises_error(self):
        """validate_tap_count should raise ValueError for non-multiple taps."""
        from generate_filter import validate_tap_count

        # Float ratio will cause non-integer result, raising ValueError
        with pytest.raises(ValueError, match="倍数である必要があります"):
            validate_tap_count(1024, 16.5)

    def test_zero_ratio_raises_error(self):
        """validate_tap_count should raise error for zero ratio."""
        from generate_filter import validate_tap_count

        with pytest.raises((ValueError, ZeroDivisionError)):
            validate_tap_count(1024, 0)


class TestCalculateSafeGain:
    """Tests for calculate_safe_gain function.

    #260: グローバル安全ゲインの算出。max_coef_max > 1.0 の場合に
    gain = M / max_coef_max を計算してクリッピングを防止する。
    """

    def test_calculates_from_max_coef(self, tmp_path):
        """max_coef > 1.0 の場合は max_coef ベースで計算"""
        from generate_filter import calculate_safe_gain

        # テスト用JSONファイルを作成
        (tmp_path / "filter_test.json").write_text(
            '{"validation_results": {"normalization": {"l1_norm": 100.0, "max_coefficient_amplitude": 1.5}}}'
        )

        filter_infos = [
            ("test", "filter_test", 1000, {"input_rate": 44100, "ratio": 16})
        ]
        result = calculate_safe_gain(
            filter_infos, safety_margin=0.97, coefficients_dir=str(tmp_path)
        )

        # H = 0.97 / 1.5 ≈ 0.6467
        assert result["max_coef_max"] == 1.5
        assert np.isclose(result["recommended_gain"], 0.97 / 1.5, rtol=1e-6)
        assert result["recommended_gain"] < 1.0

    def test_returns_1_when_all_safe(self, tmp_path):
        """全フィルタの max_coef <= 1.0 の場合は gain=1.0"""
        from generate_filter import calculate_safe_gain

        # max_coef が 1.0 以下のJSONファイル
        (tmp_path / "filter_safe.json").write_text(
            '{"validation_results": {"normalization": {"l1_norm": 50.0, "max_coefficient_amplitude": 0.8}}}'
        )

        filter_infos = [
            ("safe", "filter_safe", 1000, {"input_rate": 44100, "ratio": 16})
        ]
        result = calculate_safe_gain(
            filter_infos, safety_margin=0.97, coefficients_dir=str(tmp_path)
        )

        # max_coef=0.8 → H = 0.97/0.8 = 1.2125 → clamp to 1.0
        assert result["max_coef_max"] == 0.8
        assert result["recommended_gain"] == 1.0

    def test_handles_multiple_filters(self, tmp_path):
        """複数フィルタから最大値を正しく取得"""
        from generate_filter import calculate_safe_gain

        (tmp_path / "filter_a.json").write_text(
            '{"validation_results": {"normalization": {"l1_norm": 100.0, "max_coefficient_amplitude": 0.9}}}'
        )
        (tmp_path / "filter_b.json").write_text(
            '{"validation_results": {"normalization": {"l1_norm": 200.0, "max_coefficient_amplitude": 1.2}}}'
        )

        filter_infos = [
            ("a", "filter_a", 1000, {}),
            ("b", "filter_b", 1000, {}),
        ]
        result = calculate_safe_gain(
            filter_infos, safety_margin=0.97, coefficients_dir=str(tmp_path)
        )

        # max_coef_max = 1.2, l1_max = 200.0
        assert result["max_coef_max"] == 1.2
        assert result["max_coef_max_filter"] == "b"
        assert result["l1_max"] == 200.0
        assert result["l1_max_filter"] == "b"
        assert np.isclose(result["recommended_gain"], 0.97 / 1.2, rtol=1e-6)

    def test_handles_invalid_data_gracefully(self, tmp_path):
        """無効なデータ（None等）を安全にスキップ"""
        from generate_filter import calculate_safe_gain

        # l1_norm が null のJSON
        (tmp_path / "filter_invalid.json").write_text(
            '{"validation_results": {"normalization": {"l1_norm": null, "max_coefficient_amplitude": null}}}'
        )
        (tmp_path / "filter_valid.json").write_text(
            '{"validation_results": {"normalization": {"l1_norm": 50.0, "max_coefficient_amplitude": 0.5}}}'
        )

        filter_infos = [
            ("invalid", "filter_invalid", 1000, {}),
            ("valid", "filter_valid", 1000, {}),
        ]
        result = calculate_safe_gain(
            filter_infos, safety_margin=0.97, coefficients_dir=str(tmp_path)
        )

        # 無効なフィルタはスキップされ、有効なフィルタのみ処理される
        assert len(result["details"]) == 1
        assert result["details"][0]["name"] == "valid"
        assert result["max_coef_max"] == 0.5

    def test_handles_missing_json_file(self, tmp_path):
        """存在しないJSONファイルをスキップ"""
        from generate_filter import calculate_safe_gain

        filter_infos = [("missing", "filter_missing", 1000, {})]
        result = calculate_safe_gain(
            filter_infos, safety_margin=0.97, coefficients_dir=str(tmp_path)
        )

        # ファイルがないので details は空
        assert len(result["details"]) == 0
        assert result["recommended_gain"] == 1.0  # デフォルト値

    def test_int_float_conversion_safety(self, tmp_path):
        """int型の値も正しくfloatに変換される"""
        from generate_filter import calculate_safe_gain

        # JSONでは整数として保存されることがある
        (tmp_path / "filter_int.json").write_text(
            '{"validation_results": {"normalization": {"l1_norm": 100, "max_coefficient_amplitude": 1}}}'
        )

        filter_infos = [("int_test", "filter_int", 1000, {})]
        result = calculate_safe_gain(
            filter_infos, safety_margin=0.97, coefficients_dir=str(tmp_path)
        )

        # int でも float として処理される
        assert isinstance(result["l1_max"], float)
        assert isinstance(result["max_coef_max"], float)
        assert result["l1_max"] == 100.0
        assert result["max_coef_max"] == 1.0
        # max_coef=1.0 → H = 0.97 / 1.0 = 0.97 < 1.0
        assert np.isclose(result["recommended_gain"], 0.97, rtol=1e-6)
