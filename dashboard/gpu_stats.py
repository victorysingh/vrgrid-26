"""Live GPU telemetry for the dashboard -- no Rerun dependency. [JP]

The map's pixels are drawn on the GPU: on the demo laptop Rerun picks the
discrete "NVIDIA GeForce RTX 4050 Laptop GPU" over the integrated Radeon (its
debug log says so). This module reads what that GPU is doing, so the dashboard
can show it, and nothing more.

What the panel shows depends on how the pipeline was started. With the
default `--device cpu` the map is computed in NumPy on the CPU, and the GPU is
only rendering -- a panel implying otherwise would be false. With
`--device cuda` (`python -m vrgrid.dash` or `python -m vrgrid.run`) perception
and every map stage except Patchwork++ run on the card
(`docs/gpu-lane/08-GPU-FRAME-LOOP.md`), and the utilisation and memory read
here include that work. The reading is the whole card either way; it does not
separate the pipeline from the renderer. The chart title says "at record time"
because a baked recording can only carry what the GPU was doing while it was
being made.

`nvidia-smi` costs 75-270 ms a call (measured here), far too slow for a 100 ms
frame, so a daemon thread polls it about once a second and the dashboard reads
the latest snapshot. No NVIDIA GPU, no `nvidia-smi`, or a failed call: every
reading is None and the dashboard simply leaves the GPU rows out.
"""

import shutil
import subprocess
import threading
from dataclasses import dataclass

QUERY = ["--query-gpu=name,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"]


@dataclass(frozen=True)
class GpuReading:
    name: str
    util_pct: float
    mem_used_mib: float
    mem_total_mib: float

    @property
    def mem_pct(self) -> float:
        return 100.0 * self.mem_used_mib / self.mem_total_mib if self.mem_total_mib else 0.0


def parse_nvidia_smi(text: str) -> GpuReading | None:
    """First well-formed `name, util, mem used, mem total` line, or None.

    e.g. "NVIDIA GeForce RTX 4050 Laptop GPU, 0, 18, 6141"."""
    for line in (text or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            return GpuReading(parts[0], float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            continue
    return None


class GpuSampler:
    """Background poller of `nvidia-smi`. `latest()` never blocks on the GPU."""

    def __init__(self, interval_s: float = 1.0, exe: str | None = None):
        self.exe = exe if exe is not None else shutil.which("nvidia-smi")
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = None
        if self.exe:
            self._latest = self._query()          # one synchronous read: a name for frame 0
            threading.Thread(target=self._loop, daemon=True, name="gpu-sampler").start()

    def _query(self) -> GpuReading | None:
        try:
            out = subprocess.run([self.exe, *QUERY], capture_output=True, text=True, timeout=5,
                                 check=False,      # the return code is checked below
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            return None
        return parse_nvidia_smi(out.stdout) if out.returncode == 0 else None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            reading = self._query()
            if reading is not None:
                with self._lock:
                    self._latest = reading

    def latest(self) -> GpuReading | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
