"""GPU telemetry for the dashboard. [JP]

`dashboard/gpu_stats.py` imports no rerun and no GPU library, so these run in CI
on machines with no GPU at all -- which is exactly the case they must handle.
"""

import pytest
from vrgrid.dash.gpu_stats import GpuReading, GpuSampler, parse_nvidia_smi


def test_parses_the_line_nvidia_smi_prints():
    r = parse_nvidia_smi("NVIDIA GeForce RTX 4050 Laptop GPU, 37, 1536, 6141\n")
    assert r == GpuReading("NVIDIA GeForce RTX 4050 Laptop GPU", 37.0, 1536.0, 6141.0)
    assert r.mem_pct == pytest.approx(100 * 1536 / 6141)


@pytest.mark.parametrize("text", ["", "garbage", "name, not-a-number, 1, 2", "a, 1, 2"])
def test_anything_else_is_no_reading_rather_than_an_error(text):
    assert parse_nvidia_smi(text) is None


def test_the_first_good_line_wins_on_a_multi_gpu_machine():
    text = "bad line\nGPU A, 10, 100, 1000\nGPU B, 90, 900, 1000\n"
    assert parse_nvidia_smi(text).name == "GPU A"


def test_zero_total_memory_does_not_divide_by_zero():
    assert GpuReading("x", 0.0, 0.0, 0.0).mem_pct == 0.0


def test_no_nvidia_smi_means_no_readings_and_no_crash():
    sampler = GpuSampler(interval_s=0.05, exe="C:/definitely/not/nvidia-smi.exe")
    try:
        assert sampler.latest() is None
    finally:
        sampler.stop()


def test_an_empty_exe_path_starts_no_thread():
    sampler = GpuSampler(exe="")
    assert sampler.latest() is None
    sampler.stop()
