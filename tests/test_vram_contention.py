"""The VRAM / contention report renders from its recorded measurement. [Shrestha]

The measurement itself needs a card and minutes of GPU time, so CI does not
run it. What CI can check is that the committed JSON still carries every field
the report and `docs/gpu-lane/09-VRAM-CONTENTION.md` read, and that the
attribution adds up: the declared device total is what cupy's pool holds.
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/gpu-lane/09-vram-contention.json"


def _script():
    spec = importlib.util.spec_from_file_location("vram_contention",
                                                  ROOT / "scripts/vram_contention.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_recorded_measurement_renders(capsys):
    _script().report(json.loads(RECORD.read_text()))
    out = capsys.readouterr().out
    assert "VRAM attribution" in out and "Contention" in out


def test_the_declared_device_memory_is_what_the_pool_holds():
    g = json.loads(RECORD.read_text())["grid"]
    declared = g["map_static_bytes"] + g["perception_bytes"]
    assert abs(declared - g["cupy_pool_used"]) < 0.001 * declared
    assert g["cupy_pool_used"] <= g["cupy_pool_reserved"]


def test_every_timed_configuration_kept_its_frames():
    o = json.loads(RECORD.read_text())
    frames = o["env"]["frames"]
    for cfg in ("grid", "frnet", "both"):
        assert o[cfg]["frame_ms"]["n"] == frames, cfg
    for cfg in ("grid", "frnet"):
        assert o["pair"][cfg]["frame_ms"]["n"] > 0, cfg
