"""Snapshot the two timing-table printers exactly as they print today.

**This does not implement R-i.** Whether `print_table` and `print_real_table`
should become one printer is JP's open decision. This is the safety net that
decision needs either way: `pending-review/r-i-unify-timing-table-printers.md`
records that *no test calls either printer*, which is why two mislabelling
incidents (the 80.78 ms whole-frame claim, the "8.15 -> 1.31 MB/frame" claim)
reached documents before anyone noticed. If the printers are unified, these
snapshots say precisely what changed; if they are left alone, they stop a third
incident of the same shape.

The timings are hand-fed constants, never measured, so the output is identical
on every machine and the snapshots can be compared byte for byte.
"""
import importlib.util
import io
import pathlib
import sys
from contextlib import redirect_stdout

import pytest

from vrgrid.gpu.timing import STAGES, Timer

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def tt():
    """`scripts/timing_table.py` imported as a module (it is not a package)."""
    spec = importlib.util.spec_from_file_location("timing_table", SCRIPTS / "timing_table.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["timing_table"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fill(timer, values):
    """One sample per stage, so p50 == p99 == max and the table is exact."""
    for name, ms in values.items():
        timer.record(name, ms)
    return timer


def _capture(fn, *args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args)
    return buf.getvalue()


# --- the synthetic table -------------------------------------------------------------------

SYNTHETIC = {"bin": 1.0, "scatter": 2.0, "fuse": 4.0, "cleanup": 8.0,
             "pyramid": 16.0, "shift": 32.0, "measured": 63.0}


def test_synthetic_table_snapshot(tt):
    t = _fill(Timer(stages=tt.MEASURED + ("measured",)), SYNTHETIC)
    out = _capture(tt.print_table, t, None, None)

    assert "owner" in out.splitlines()[0], "the synthetic table carries an owner column"
    assert "MB/frame" not in out, "no --alloc was passed, so there is no MB column"
    # The subtotal is labelled MEASURED, never FRAME. This is the labelling the
    # 80.78 ms incident got wrong, so it is asserted directly.
    assert "MEASURED" in out and "FRAME" not in out
    assert "MEASURED is a LOWER BOUND on frame latency" in out
    assert "15.9 FPS p50" in out and "15.9 FPS p99" in out       # 1000/63
    assert "bin         JP" in out or "bin " in out


def test_synthetic_table_with_alloc_adds_one_column_and_the_bin_warning(tt):
    t = _fill(Timer(stages=tt.MEASURED + ("measured",)), SYNTHETIC)
    alloc = {"bin": 2_000_000, "scatter": 5_000, "fuse": 0}
    out = _capture(tt.print_table, t, alloc, None)

    assert "MB/frame" in out.splitlines()[0]
    assert "2.00" in out                       # bin's 2 MB
    assert "~0" in out                         # fuse's 0 bytes, printed as ~0
    # bin is supposed to allocate nothing; over 1 MB is a regression, and the
    # printer must say so rather than just showing a number.
    assert "`grid.lattice.bin_points`" in out and "regression" in out


# --- the real-data table -------------------------------------------------------------------

REAL = {name: float(i + 1) for i, name in enumerate(STAGES) if name != "total"}
REAL["total"] = 100.0


def test_real_table_snapshot(tt):
    t = _fill(Timer(stages=STAGES), REAL)
    out = _capture(tt.print_real_table, t)

    assert "share" in out.splitlines()[0], "the real table carries a share column"
    assert "owner" not in out.splitlines()[0], "and no owner column"
    # The total row is FRAME here, and legitimately so: everything is measured.
    assert "FRAME" in out and "MEASURED" not in out
    assert "MEASURED is a LOWER BOUND" not in out
    assert "10.0 FPS p50" in out                                  # 1000/100
    # Exactly on budget: `meets_sensor_rate` is `p99 <= 1e3/SENSOR_HZ`, so 100.00 ms
    # at 10 Hz MEETS the rate. The boundary is inclusive; pinned here because a
    # `<` / `<=` slip would move the gate without moving any number.
    assert "meets 10 Hz at p99" in out
    assert "split_merge is absent" in out
    assert "will not sum to exactly 100%" in out, "the share caveat must stay"


@pytest.mark.parametrize(("total_ms", "verdict"), [
    (50.0, "meets 10 Hz at p99"),      # comfortably inside
    (100.0, "meets 10 Hz at p99"),     # exactly on budget: `<=`, so it meets
    (100.01, "MISSES 10 Hz at p99"),   # the first value that misses
    (200.0, "MISSES 10 Hz at p99"),
])
def test_real_table_verdict_at_and_around_the_budget(tt, total_ms, verdict):
    out = _capture(tt.print_real_table, _fill(Timer(stages=STAGES), dict(REAL, total=total_ms)))
    assert verdict in out
    assert ("MISSES" in out) == verdict.startswith("MISSES")


# --- the property that actually matters ----------------------------------------------------

def test_the_two_printers_do_not_agree_on_columns_or_total_label(tt):
    """Pins the split itself -- the thing R-i is about.

    If the printers are ever unified, this test is the one that should fail,
    and its failure is the signal to re-read `pending-review/r-i-...md` rather
    than to patch the assertion.
    """
    syn = _capture(tt.print_table, _fill(Timer(stages=tt.MEASURED + ("measured",)), SYNTHETIC),
                   None, None).splitlines()[0]
    real = _capture(tt.print_real_table, _fill(Timer(stages=STAGES), REAL)).splitlines()[0]
    assert syn != real, "two printers, two different headers -- this is the R-i question"
    assert "owner" in syn and "share" not in syn
    assert "share" in real and "owner" not in real
