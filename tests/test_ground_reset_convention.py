"""Enforce D1 option A: every direct ground caller resets the estimator per run.

JP decided D1 as **option A** (2026-09-18): `ground.py` keeps its module-level
Patchwork++ estimator, and every entry point calls `ground.reset_estimator()`
once per run. See `pending-review/d1-patchworkpp-estimator-lifetime.md`.

A's one weakness, stated when it was chosen: *a convention holds only while
every caller remembers it, and no test failed when one forgot.* This file is
that missing test. It does not change `ground.py` and does not implement B or C.

Two guards:

1. **Static.** Any module that calls `segment_ground` / `segment_ground_or_fallback`
   directly drives the estimator itself and must also manage its lifetime. Found
   by walking the AST, not by a hand-maintained list, so a NEW entry point that
   forgets is caught the day it is added.
2. **Behavioural.** `iter_pipeline` must reset exactly ONCE per run -- not per
   frame. Per-frame would be correct but would move a Patchwork++ construction
   into the timed path and silently change every latency number.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEARCH = ("src", "scripts", "dashboard", "reports/harnesses")

# `ground.py` defines these; it is not a caller of them.
EXEMPT = {ROOT / "src" / "perception" / "ground.py"}

# Found by this test on its first run, 2026-09-18. NOT silently allowed, and not
# silently fixed either: `ground_cost.py` is a committed provenance harness that
# produced a published figure (R-a's isolated ground cost, 21.01 ms p50, quoted
# again in D8's 4.4x variance note). It warms up with one `segment_ground` call
# and then benches `reps=3` over the same scans without ever resetting, so reps 2
# and 3 start from state rep 1 left behind -- which a real run never does.
# Editing the harness would change the provenance of a published number, so it is
# logged as its own open item (OPEN-ITEMS R-k) for a deliberate pass.
# This list must not grow: a NEW file here means a new entry point forgot.
KNOWN_UNRESET = {"reports/harnesses/ground_cost.py"}

DRIVES_GROUND = {"segment_ground", "segment_ground_or_fallback"}
MANAGES_LIFETIME = "reset_estimator"
PRIVATE_GLOBAL = "_estimator"


def _called_names(tree):
    """Every attribute/name actually CALLED in the module."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                out.add(f.attr)
            elif isinstance(f, ast.Name):
                out.add(f.id)
    return out


def _assigns_private_estimator(tree):
    """`ground._estimator = ...` -- the documented escape hatch for a harness
    that must install its OWN configured estimator (num_iter work). Option A
    keeps this legal because the public API cannot express it."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == PRIVATE_GLOBAL:
                    return True
    return False


def _modules():
    for rel in SEARCH:
        for path in sorted((ROOT / rel).rglob("*.py")):
            if path in EXEMPT or "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):      # not ours to police
                continue
            yield path, tree


def _direct_ground_callers():
    for path, tree in _modules():
        if _called_names(tree) & DRIVES_GROUND:
            yield path, tree


def test_the_scan_finds_callers_at_all():
    """Guards the guard: a broken scanner would make every test below pass
    vacuously, which is exactly the failure mode this file exists to prevent."""
    found = [p for p, _ in _direct_ground_callers()]
    assert len(found) >= 4, f"scanner found only {found} -- it has stopped working"


def test_every_direct_ground_caller_manages_the_estimator_lifetime():
    """D1 option A's invariant, enforced.

    If this fails on a file you just added: call `ground.reset_estimator()` once
    per run, before the frame loop -- not per frame. Patchwork++ adapts its
    thresholds from the scans it has seen, so a shared estimator makes a second
    run in the same process produce a different map.
    """
    offenders = []
    for path, tree in _direct_ground_callers():
        if MANAGES_LIFETIME in _called_names(tree):
            continue
        if _assigns_private_estimator(tree):
            continue
        offenders.append(str(path.relative_to(ROOT)).replace("\\", "/"))

    new = sorted(set(offenders) - KNOWN_UNRESET)
    assert not new, (
        "these call segment_ground*() but never reset the shared Patchwork++ "
        f"estimator: {new}. See pending-review/"
        "d1-patchworkpp-estimator-lifetime.md (D1 was decided as option A)."
    )


def test_the_known_exception_list_has_not_gone_stale():
    """If someone fixes `ground_cost.py`, this says so, so the list shrinks.

    An allowlist nobody prunes becomes a place defects go to be forgotten.
    """
    offenders = set()
    for path, tree in _direct_ground_callers():
        names = _called_names(tree)
        if MANAGES_LIFETIME not in names and not _assigns_private_estimator(tree):
            offenders.add(str(path.relative_to(ROOT)).replace("\\", "/"))

    fixed = sorted(KNOWN_UNRESET - offenders)
    assert not fixed, f"these now reset and should leave KNOWN_UNRESET: {fixed}"


def test_the_known_entry_points_are_still_the_public_kind():
    """The five entry points use the PUBLIC call, not the private global.

    Harnesses may poke `ground._estimator` (option A accepts that cost, because
    installing a differently configured estimator has no public equivalent).
    Shipped entry points may not: they are the ones a new contributor copies.
    """
    shipped = {"src/run/__main__.py", "src/eval/harness.py", "scripts/feature_report.py",
               "scripts/gen_demo_rrds.py", "scripts/gpu_parity.py"}
    seen = {}
    for path, tree in _direct_ground_callers():
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if rel in shipped:
            seen[rel] = MANAGES_LIFETIME in _called_names(tree)

    assert set(seen) == shipped, f"entry points moved or vanished: {sorted(seen)}"
    assert all(seen.values()), f"not using the public reset: {[k for k, v in seen.items() if not v]}"


@pytest.mark.determinism
def test_iter_pipeline_resets_once_per_run_not_once_per_frame(monkeypatch):
    """Once per run keeps the reset out of the timed per-frame path.

    Moving it inside the loop would still be deterministic, so no other test
    would notice -- it would just quietly add a Patchwork++ construction to
    every frame and change every latency figure on the branch.
    """
    loader = pytest.importorskip("vrgrid.perception.loader")
    if not (loader.verify_sequence_exists("08") and loader._velodyne_path("08", 0).exists()):
        pytest.skip("KITTI seq 08 not present -- set VRGRID_DATA_ROOT")

    from vrgrid.perception import ground
    from vrgrid.run.__main__ import iter_pipeline

    calls = []
    real = ground.reset_estimator
    monkeypatch.setattr(ground, "reset_estimator", lambda: (calls.append(1), real())[1])

    frames = list(iter_pipeline("08", 3))
    assert len(frames) == 3
    assert len(calls) == 1, f"reset_estimator ran {len(calls)}x for 3 frames; expected once per run"
