"""`paired_stats` in `scripts/plan_regret_frnet_delta.py`, pinned without data or a model.

The D11 plan-regret delta was reported with a rough UNPAIRED standard error. The two
arms answer the SAME 64 queries, so the honest uncertainty is over per-query
DIFFERENCES. These tests pin that arithmetic: the mean difference equals the
difference of means, the SE comes from the SD of the differences, the bootstrap
resamples queries (not arms independently), and the sign test is exact.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plan_regret_frnet_delta import paired_stats  # noqa: E402


def test_mean_difference_equals_difference_of_means_and_se_uses_the_differences():
    rng = np.random.default_rng(3)
    gt = rng.uniform(0, 3, 64)
    fr = gt + rng.normal(0.5, 0.2, 64)
    s = paired_stats(gt, fr, n_boot=2000, seed=0)
    d = fr - gt
    assert s["n"] == 64
    assert s["mean_diff"] == pytest.approx(fr.mean() - gt.mean(), abs=1e-12)
    assert s["sd_diff"] == pytest.approx(d.std(ddof=1), abs=1e-12)
    assert s["se_diff"] == pytest.approx(d.std(ddof=1) / math.sqrt(64), abs=1e-12)
    lo, hi = s["boot_ci95"]
    assert lo < s["mean_diff"] < hi


def test_identical_arms_give_exactly_zero_and_all_ties():
    """The oracle control, reduced to arithmetic: every difference is 0."""
    v = np.linspace(0.0, 2.0, 64)
    s = paired_stats(v, v.copy(), n_boot=500, seed=0)
    assert s["mean_diff"] == 0.0 and s["sd_diff"] == 0.0
    assert (s["worse"], s["equal"], s["better"]) == (0, 64, 0)
    assert s["sign_test_p"] == 1.0
    assert s["boot_ci95"] == [0.0, 0.0]


def test_sign_test_is_the_exact_two_sided_binomial():
    gt = np.zeros(10)
    fr = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, -1], dtype=float)   # 9 worse, 1 better
    s = paired_stats(gt, fr, n_boot=200, seed=0)
    expected = min(1.0, 2 * sum(math.comb(10, k) for k in range(0, 2)) / 2 ** 10)
    assert (s["worse"], s["better"]) == (9, 1)
    assert s["sign_test_p"] == pytest.approx(expected, abs=1e-15)


def test_bootstrap_is_seeded_and_reproducible():
    rng = np.random.default_rng(5)
    gt, fr = rng.uniform(0, 1, 40), rng.uniform(0, 1, 40)
    assert paired_stats(gt, fr, n_boot=1000, seed=7) == paired_stats(gt, fr, n_boot=1000, seed=7)


def test_misaligned_arms_are_refused():
    with pytest.raises(ValueError):
        paired_stats(np.zeros(5), np.zeros(6))
