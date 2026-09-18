"""Session-wide test isolation."""

import pytest


class _NoGpu:
    """Stands in for dashboard.gpu_stats.GpuSampler: no reading, no thread."""

    def latest(self):
        return None

    def stop(self):
        pass


@pytest.fixture(autouse=True)
def _no_gpu_polling_thread(monkeypatch):
    """No real `nvidia-smi` poller in any test.

    `PipelineView` starts one and only `finish()` stops it. Tests across
    test_dashboard, test_dashboard_features, test_run, test_cvd and
    test_dense3d_comparison build views without finishing them, so their daemon
    threads kept polling for the rest of the session -- and the tracemalloc
    tests (test_determinism, test_timing, test_visibility), which trace every
    thread, counted those calls as allocation, failing only in a full run.
    The sampler itself is tested directly in tests/test_gpu_stats.py.
    """
    try:
        from vrgrid.dash import pipeline_view
    except ImportError:              # no rerun installed: no view, no poller
        return
    monkeypatch.setattr(pipeline_view, "GpuSampler", _NoGpu)
