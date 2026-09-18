#!/usr/bin/env python3
"""What is actually installed, and whether it works. [Shrestha]

Run after `scripts/setup_env.sh`, and on any machine before quoting a number
from it. Every line is a capability the pipeline depends on, checked by using
it rather than by importing it -- cupy in particular imports cleanly and then
fails at the first kernel launch when its CUDA headers are missing, which is
the failure that looks like a broken GPU.

Exit status is 0 when everything the pipeline REQUIRES is present. GPU and
dataset are reported but never fatal: a CPU-only box is a supported
configuration and CI is one.
"""
import importlib.metadata as md
import os
import sys

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "
required_failed = False


def line(status, name, detail=""):
    print(f"[{status}] {name:<22} {detail}")


def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


print(f"python                 {sys.version.split()[0]}")
if sys.version_info[:2] > (3, 13):
    line(WARN, "python version", f"{sys.version_info.major}.{sys.version_info.minor}: "
         "pypatchworkpp has no wheel above 3.13 and will build from source")

# --- required ---------------------------------------------------------------
for pkg in ("numpy", "pyyaml", "pytest"):
    v = version(pkg)
    if v:
        line(OK, pkg, v)
    else:
        line(BAD, pkg, "missing")
        required_failed = True

try:
    import vrgrid.grid.lattice  # noqa: F401
    from vrgrid.gpu.kernels import new_sorted_scratch, scatter_sorted  # noqa: F401
    line(OK, "vrgrid", "importable")
except Exception as exc:                                          # noqa: BLE001
    line(BAD, "vrgrid", f"{type(exc).__name__}: {exc}")
    required_failed = True

# --- ground segmentation ----------------------------------------------------
pv = version("pypatchworkpp")
if pv:
    line(OK if pv == "1.4.1" else WARN, "pypatchworkpp",
         pv + ("" if pv == "1.4.1" else "  EXPECTED 1.4.1 -- see pyproject.toml"))
else:
    line(WARN, "pypatchworkpp", "absent: ground falls back to the semantic proxy, "
                                "and every rho/curb number is on the fallback")

# --- gpu --------------------------------------------------------------------
try:
    import cupy
    try:
        a = cupy.arange(8, dtype=cupy.int32)
        assert int(a.sum()) == 28
        name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        cc = cupy.cuda.Device(0).compute_capability
        line(OK, "cupy", f"{cupy.__version__}  {name} sm_{cc}")
    except Exception as exc:                                      # noqa: BLE001
        line(WARN, "cupy", f"imports but cannot run a kernel: {type(exc).__name__}. "
                           "CUDA_PATH? see scripts/setup_env.sh")
except ImportError:
    line(WARN, "cupy", "absent: GPU kernels and their tests will skip")

try:
    import torch
    line(OK if torch.cuda.is_available() else WARN, "torch",
         f"{torch.__version__}  cuda={torch.cuda.is_available()}")
except ImportError:
    line(WARN, "torch", "absent: FRNet eval and fine-tune will not run")

# --- data -------------------------------------------------------------------
root = os.environ.get("VRGRID_DATA_ROOT")
if not root:
    line(WARN, "VRGRID_DATA_ROOT", "unset: data-backed tests skip (they do NOT fail)")
elif not os.path.isdir(os.path.join(root, "sequences")):
    line(WARN, "VRGRID_DATA_ROOT", f"{root}: no sequences/ under it")
else:
    seqs = sorted(os.listdir(os.path.join(root, "sequences")))
    line(OK, "VRGRID_DATA_ROOT", f"{len(seqs)} sequences, {seqs[0]}..{seqs[-1]}")

ckpt = os.environ.get("VRGRID_FRNET_CHECKPOINT")
if not ckpt:
    line(WARN, "FRNET_CHECKPOINT", "unset: FRNet eval will not run")
else:
    line(OK if os.path.isfile(ckpt) else WARN, "FRNET_CHECKPOINT", ckpt)

print()
if required_failed:
    print("REQUIRED components are missing -- the pipeline will not run.")
    sys.exit(1)
print("Required components present. Warnings above are optional capabilities;")
print("each one that is absent silently narrows what the numbers mean.")
