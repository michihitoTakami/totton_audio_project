import math

import numpy as np
import pytest

from scripts import check_headroom


def test_compute_filter_stats_detects_peak():
    coeffs = np.array([0.5, -1.2, 0.1], dtype=np.float32)
    stats = check_headroom.compute_filter_stats(coeffs)
    assert stats["taps"] == 3
    assert stats["max_linear"] == pytest.approx(1.2)
    assert stats["max_dbfs"] == pytest.approx(20 * math.log10(1.2))
    assert stats["required_headroom_db"] == pytest.approx(20 * math.log10(1.2))


def test_analyze_buffer_reports_clipping():
    buffer = np.array([[0.2, -0.3], [1.1, 0.05], [-0.9, 1.01]], dtype=np.float32)
    stats = check_headroom.analyze_buffer(buffer, bins=8)

    assert stats["channels"] == 2
    assert stats["total_samples"] == buffer.size
    assert stats["peak_linear"] == pytest.approx(1.1)
    assert stats["clip_count"] == 2
    assert stats["per_channel_peaks"][0] == pytest.approx(1.1)
    assert stats["per_channel_peaks"][1] == pytest.approx(1.01)


