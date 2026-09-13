# PROVENANCE -- written 2026-09-14 under OPEN-ITEMS.md item D8 (scope upgraded).
#
# Produced: the machine-state record attached to every latency figure from
#           2026-09-14 onward.
# Run:      python reports/harnesses/machine_state.py
#
# NOTE: D8 was upgraded after this laptop gave 21 ms and 93 ms for the same call
#      within one session -- CPU downclocked to 1520/2400 MHz and ~8 GB of memory
#      over-committed. Neither is visible in a timing table. So every timing run
#      records this, before and after, and a run whose state is out of bounds is
#      reported as such rather than quietly averaged in.
#
# Measurement only. Changes nothing.
"""Snapshot the machine state that decides whether a timing number means anything."""
import json
import platform
import subprocess
import sys

_PS = (
    "$p = Get-CimInstance Win32_Processor | Select-Object -First 1; "
    "$os = Get-CimInstance Win32_OperatingSystem; "
    "[pscustomobject]@{"
    "cpu = $p.Name.Trim(); "
    "clock_mhz = [int]$p.CurrentClockSpeed; "
    "max_clock_mhz = [int]$p.MaxClockSpeed; "
    "load_pct = [int]$p.LoadPercentage; "
    "phys_gb = [math]::Round($os.TotalVisibleMemorySize/1MB, 2); "
    "free_gb = [math]::Round($os.FreePhysicalMemory/1MB, 2); "
    "commit_gb = [math]::Round(($os.TotalVirtualMemorySize - $os.FreeVirtualMemory)/1MB, 2); "
    "uptime_min = [int]((Get-Date) - $os.LastBootUpTime).TotalMinutes"
    "} | ConvertTo-Json -Compress"
)

# Bounds a run must satisfy to be trusted. Derived from the 2026-09-13 failure,
# not chosen for convenience: 1520/2400 MHz (63%) and 23.6 GB committed on 15.7 GB
# physical produced a 4.4x slowdown.
MIN_CLOCK_FRACTION = 0.95
MAX_COMMIT_FRACTION = 1.00


def snapshot() -> dict:
    s = {"python": platform.python_version(), "platform": platform.platform()}
    if sys.platform != "win32":
        s["note"] = "state capture implemented for Windows only"
        s["trusted"] = None
        return s
    out = subprocess.run(["powershell", "-NoProfile", "-Command", _PS],
                         capture_output=True, text=True, timeout=60)
    s.update(json.loads(out.stdout))
    reasons = []
    if s["clock_mhz"] < MIN_CLOCK_FRACTION * s["max_clock_mhz"]:
        reasons.append(f"CPU clock {s['clock_mhz']}/{s['max_clock_mhz']} MHz is below "
                       f"{MIN_CLOCK_FRACTION:.0%} of base")
    if s["commit_gb"] > MAX_COMMIT_FRACTION * s["phys_gb"]:
        reasons.append(f"commit {s['commit_gb']} GB exceeds physical {s['phys_gb']} GB "
                       "(the machine is paging)")
    s["trusted"] = not reasons
    s["untrusted_because"] = reasons
    return s


def line(s: dict) -> str:
    if s.get("trusted") is None:
        return f"state: {s.get('note')}"
    flag = "OK" if s["trusted"] else "UNTRUSTED: " + "; ".join(s["untrusted_because"])
    return (f"clock {s['clock_mhz']}/{s['max_clock_mhz']} MHz, load {s['load_pct']}%, "
            f"commit {s['commit_gb']}/{s['phys_gb']} GB, free {s['free_gb']} GB, "
            f"uptime {s['uptime_min']} min -- {flag}")


if __name__ == "__main__":
    s = snapshot()
    print(json.dumps(s, indent=2))
    print(line(s))
