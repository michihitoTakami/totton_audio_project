"""
Unit tests for scripts/generate_mixed_phase.py
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_filter import FilterConfig  # noqa: E402
from generate_mixed_phase import (  # noqa: E402
    GroupDelayAnalysis,
    MixedPhaseGenerator,
    MixedPhaseSettings,
    PhaseEqDesigner,
    analyze_base_filter,
    build_total_delay_target,
    build_weight_profile,
    measure_group_delay,
    smooth_curve,
    summarize_group_delay,
)


class TestMixedPhaseSettings:
    """MixedPhaseSettings のバリデーションテスト"""

    def test_default_settings_valid(self):
        """デフォルト設定は valid である"""
        settings = MixedPhaseSettings()
        # Should not raise
        settings.validate(fs=705600)

    def test_delay_samples_calculation(self):
        """delay_samples が正しく計算される"""
        settings = MixedPhaseSettings(eq_delay_ms=3.0)
        fs = 705600
        expected = 3.0 * 1e-3 * fs  # 2116.8 samples
        assert pytest.approx(settings.delay_samples(fs)) == expected

    def test_invalid_eq_low_high(self):
        """eq_low_hz >= eq_high_hz でエラー"""
        settings = MixedPhaseSettings(eq_low_hz=500.0, eq_high_hz=400.0)
        with pytest.raises(ValueError, match="eq_low_hz < eq_high_hz"):
            settings.validate(fs=705600)

    def test_eq_max_freq_above_nyquist(self):
        """eq_max_freq が Nyquist を超えるとエラー"""
        settings = MixedPhaseSettings(eq_max_freq=400_000.0)
        with pytest.raises(ValueError, match="below Nyquist"):
            settings.validate(fs=705600)  # Nyquist = 352800


class TestSmoothCurve:
    """smooth_curve 関数のテスト"""

    def test_smooth_flat_signal(self):
        """定数信号を smooth してもほぼ変わらない（境界除く）"""
        freqs = np.linspace(0, 10000, 1000)
        data = np.ones_like(freqs) * 5.0
        result = smooth_curve(freqs, data, width_hz=100.0)
        # 境界から離れた中央部分のみチェック
        np.testing.assert_allclose(result[10:-10], data[10:-10], atol=1e-6)

    def test_smooth_reduces_noise(self):
        """ノイズが smooth で減衰する"""
        freqs = np.linspace(0, 10000, 1000)
        data = np.sin(2 * np.pi * freqs / 100.0) + np.random.randn(len(freqs)) * 0.1
        result = smooth_curve(freqs, data, width_hz=200.0)
        # Smoothed signal should have lower variance
        assert np.std(result) < np.std(data)


class TestMeasureGroupDelay:
    """measure_group_delay 関数のテスト"""

    def test_impulse_zero_delay(self):
        """単純なインパルスは遅延ゼロ"""
        h = np.zeros(1024)
        h[0] = 1.0  # インパルス
        fs = 44100
        freqs, phase, tau = measure_group_delay(h, fs, n_fft=4096)
        # 遅延はほぼゼロ（数値誤差の範囲内）
        assert np.max(np.abs(tau)) < 1.0  # サンプル単位


class TestBuildWeightProfile:
    """build_weight_profile 関数のテスト"""

    def test_weight_profile_shape(self):
        """ウェイトプロファイルが正しい形状"""
        settings = MixedPhaseSettings(
            eq_low_hz=120.0, eq_high_hz=600.0, eq_low_cut_hz=70.0, eq_max_freq=20_000.0
        )
        freqs = np.linspace(0, 22050, 1000)
        weights = build_weight_profile(freqs, settings)
        assert weights.shape == freqs.shape
        assert np.all(weights >= 0)

    def test_weight_profile_transitions(self):
        """ウェイトが遷移領域で増加する"""
        settings = MixedPhaseSettings(eq_low_hz=120.0, eq_high_hz=600.0)
        freqs = np.linspace(0, 22050, 10000)
        weights = build_weight_profile(freqs, settings)

        # 低域はウェイトが小さい
        low_mask = freqs < settings.eq_low_hz
        # 高域はウェイトが大きい
        high_mask = freqs > settings.eq_high_hz
        if np.any(low_mask) and np.any(high_mask):
            assert np.mean(weights[low_mask]) < np.mean(weights[high_mask])


class TestBuildTotalDelayTarget:
    """build_total_delay_target 関数のテスト"""

    def test_target_delay_shape(self):
        """ターゲット遅延が正しい形状"""
        freqs = np.linspace(0, 22050, 1000)
        tau_min = np.ones_like(freqs) * 10.0  # 一定の最小位相遅延
        settings = MixedPhaseSettings(
            eq_delay_ms=3.0, eq_low_hz=120.0, eq_high_hz=600.0
        )
        fs = 705600

        tau_total = build_total_delay_target(freqs, tau_min, settings, fs)

        assert tau_total.shape == freqs.shape
        # tau_total は低域では tau_min 程度、高域で target delay に達する
        # smooth_curve により境界で値が変動するため、中央部分をチェック
        low_mask = (freqs >= 50) & (freqs <= settings.eq_low_hz)
        if np.any(low_mask):
            assert np.mean(tau_total[low_mask]) < 50.0  # 低域では小さい値

    def test_delay_increases_in_transition(self):
        """遷移領域で遅延が増加する"""
        freqs = np.linspace(0, 22050, 10000)
        tau_min = np.ones_like(freqs) * 5.0
        settings = MixedPhaseSettings(
            eq_delay_ms=3.0, eq_low_hz=120.0, eq_high_hz=600.0, eq_low_cut_hz=70.0
        )
        fs = 705600

        tau_total = build_total_delay_target(freqs, tau_min, settings, fs)

        # 低域（eq_low_hz以下）では tau_total ≈ tau_min
        low_mask = freqs <= settings.eq_low_hz
        if np.any(low_mask):
            assert np.mean(np.abs(tau_total[low_mask] - tau_min[low_mask])) < 10.0

        # 高域（eq_high_hz以上）では tau_total ≈ target delay
        high_mask = freqs >= settings.eq_high_hz
        if np.any(high_mask):
            target_samples = settings.delay_samples(fs)
            assert np.mean(tau_total[high_mask]) > target_samples * 0.8


class TestPhaseEqDesigner:
    """PhaseEqDesigner クラスのテスト"""

    def test_designer_initialization(self):
        """PhaseEqDesigner が正しく初期化される"""
        settings = MixedPhaseSettings(eq_taps=4096, eq_oversample=4)
        designer = PhaseEqDesigner(fs=705600, settings=settings)
        assert designer.settings.eq_taps == 4096
        assert designer.n_fft >= 4096 * 4
        assert len(designer.freqs) == designer.n_fft // 2 + 1

    def test_design_returns_correct_shape(self):
        """design が正しい形状の係数を返す"""
        settings = MixedPhaseSettings(eq_taps=2048, eq_iterations=100, use_gpu=False)
        designer = PhaseEqDesigner(fs=705600, settings=settings)
        target_phase = np.zeros(len(designer.freqs))
        weights = np.ones_like(target_phase)
        h_eq, diagnostics = designer.design(target_phase, weights)

        assert h_eq.shape == (2048,)
        assert "iterations" in diagnostics
        assert diagnostics["iterations"] > 0


class TestAnalyzeBaseFilter:
    """analyze_base_filter 関数のテスト"""

    def test_analyze_impulse(self):
        """単純なインパルスを解析"""
        h_min = np.zeros(1024)
        h_min[0] = 1.0
        settings = MixedPhaseSettings(use_gpu=False)
        fs = 44100

        analysis = analyze_base_filter(h_min, fs, settings)

        assert isinstance(analysis, GroupDelayAnalysis)
        assert len(analysis.freqs) > 0
        assert len(analysis.tau_min) == len(analysis.freqs)
        assert len(analysis.tau_total) == len(analysis.freqs)
        assert len(analysis.tau_eq) == len(analysis.freqs)
        assert len(analysis.phase_eq) == len(analysis.freqs)


class TestSummarizeGroupDelay:
    """summarize_group_delay 関数のテスト"""

    def test_summarize_perfect_match(self):
        """ターゲットと完全一致の場合、エラーはゼロ"""
        freqs = np.linspace(20, 20000, 1000)
        omega = 2.0 * np.pi * freqs / 44100
        tau_target = np.ones_like(freqs) * 100.0
        tau_actual = tau_target.copy()

        analysis = GroupDelayAnalysis(
            freqs=freqs,
            omega=omega,
            tau_min=tau_target,
            tau_total=tau_target,
            tau_eq=np.zeros_like(freqs),
            phase_eq=np.zeros_like(freqs),
            n_fft=4096,
            const_delay_samples=100.0,
        )

        summary = summarize_group_delay(analysis, tau_actual, fs=44100)

        assert summary["max_group_delay_error_samples"] < 1e-10
        assert summary["rms_group_delay_error_samples"] < 1e-10

    def test_summarize_with_error(self):
        """エラーがある場合、正しく計測される"""
        freqs = np.linspace(20, 20000, 1000)
        omega = 2.0 * np.pi * freqs / 44100
        tau_target = np.ones_like(freqs) * 100.0
        tau_actual = tau_target + 5.0  # +5サンプルの誤差

        analysis = GroupDelayAnalysis(
            freqs=freqs,
            omega=omega,
            tau_min=tau_target,
            tau_total=tau_target,
            tau_eq=np.zeros_like(freqs),
            phase_eq=np.zeros_like(freqs),
            n_fft=4096,
            const_delay_samples=100.0,
        )

        summary = summarize_group_delay(analysis, tau_actual, fs=44100)

        assert pytest.approx(summary["max_group_delay_error_samples"], abs=0.1) == 5.0
        assert pytest.approx(summary["rms_group_delay_error_samples"], abs=0.1) == 5.0


class TestMixedPhaseGenerator:
    """MixedPhaseGenerator クラスの統合テスト"""

    def test_generator_initialization(self):
        """MixedPhaseGenerator が正しく初期化される"""
        config = FilterConfig(
            n_taps=2048, input_rate=44100, upsample_ratio=2, output_prefix="test"
        )
        settings = MixedPhaseSettings(use_gpu=False)
        generator = MixedPhaseGenerator(config, settings)

        assert generator.config == config
        assert generator.settings == settings

    @patch("generate_mixed_phase.FilterPlotter.plot")
    @patch("generate_mixed_phase.FilterExporter.export")
    def test_generate_produces_valid_filter(
        self, mock_export: MagicMock, mock_plot: MagicMock
    ):
        """generate が有効なフィルタを生成する"""
        config = FilterConfig(
            n_taps=4096,
            input_rate=44100,
            upsample_ratio=2,
            output_prefix="test_mixed",
        )
        settings = MixedPhaseSettings(
            eq_taps=2048, eq_iterations=100, use_gpu=False, analysis_fft_exp=18
        )
        generator = MixedPhaseGenerator(config, settings)

        # Mock export to avoid file I/O
        mock_export.return_value = "test_mixed_2x_4k"

        base_name, n_taps = generator.generate(filter_name="test", skip_header=True)

        assert base_name == "test_mixed_2x_4k"
        assert n_taps == 4096
        mock_plot.assert_called_once()
        mock_export.assert_called_once()

        # Check metadata structure
        call_args = mock_export.call_args
        metadata = call_args[0][1]  # Second positional argument
        assert "generation_mode" in metadata
        assert metadata["generation_mode"] == "mixed_phase"
        assert "validation_results" in metadata
        assert "group_delay_metrics" in metadata
        assert "phase_eq_diagnostics" in metadata

    @patch("generate_mixed_phase.FilterPlotter.plot")
    @patch("generate_mixed_phase.FilterExporter.export")
    def test_generate_validates_settings(
        self, mock_export: MagicMock, mock_plot: MagicMock
    ):
        """generate が設定を検証する"""
        config = FilterConfig(n_taps=2048, input_rate=44100, upsample_ratio=2)
        # Invalid settings: eq_low_hz >= eq_high_hz
        settings = MixedPhaseSettings(eq_low_hz=500.0, eq_high_hz=400.0, use_gpu=False)
        generator = MixedPhaseGenerator(config, settings)

        with pytest.raises(ValueError, match="eq_low_hz < eq_high_hz"):
            generator.generate(filter_name="invalid")

    @patch("generate_mixed_phase.FilterPlotter.plot")
    @patch("generate_mixed_phase.FilterExporter.export")
    def test_metadata_includes_mixed_phase_diagnostics(
        self, mock_export: MagicMock, mock_plot: MagicMock
    ):
        """メタデータに混合位相診断情報が含まれる"""
        config = FilterConfig(n_taps=2048, input_rate=44100, upsample_ratio=2)
        settings = MixedPhaseSettings(
            eq_iterations=50, use_gpu=False, analysis_fft_exp=18
        )
        generator = MixedPhaseGenerator(config, settings)

        mock_export.return_value = "test"

        generator.generate(filter_name="test", skip_header=True)

        call_args = mock_export.call_args
        metadata: dict[str, Any] = call_args[0][1]

        # Check for required metadata fields
        assert "mixed_phase_settings" in metadata
        assert "group_delay_metrics" in metadata
        assert "phase_eq_diagnostics" in metadata

        # Verify nested structure
        assert "eq_delay_ms" in metadata["mixed_phase_settings"]
        assert "eq_low_hz" in metadata["mixed_phase_settings"]
        assert "iterations" in metadata["phase_eq_diagnostics"]
