#!/usr/bin/env bash
# Build the full vrgrid environment from nothing. [Shrestha]
#
# Everything the pipeline needs to start: the package, Patchwork++ ground
# segmentation, the CuPy GPU stack and Torch for FRNet. Written to be the same
# script on a laptop and on an AWS instance, because a build that differs
# between the two is how two people end up comparing different systems --
# `docs/patchwork-build-record.md` exists because that already happened once.
#
#     ./scripts/setup_env.sh              # CPU + GPU, the normal case
#     VRGRID_NO_GPU=1 ./scripts/setup_env.sh   # CPU only (CI, or no card)
#
# Then, per shell or in your profile:
#     export VRGRID_DATA_ROOT=/path/to/kitti/dataset
#     export VRGRID_FRNET_CHECKPOINT=/path/to/frnet-semantickitti_seg.pth
#
# ⚑ PYTHON 3.13, NOT NEWER. pypatchworkpp publishes wheels for CPython
#   3.8-3.13 only. On 3.14 pip falls back to the sdist and builds from source,
#   which needs cmake and a compiler and produces a binary nobody can match
#   later. 3.13 gets the pinned 1.4.1 wheel, which is what the version pin in
#   pyproject.toml is for -- see that file for why 1.4.1 and not "latest".
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${VRGRID_PYTHON:-python3.13}"
command -v "$PY" >/dev/null || { echo "need $PY on PATH (set VRGRID_PYTHON to override)"; exit 1; }

echo "==> venv (.venv) with $($PY -V)"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip -q install --upgrade pip

echo "==> vrgrid + dev + perception + dash + report"
.venv/bin/pip -q install -e ".[dev,perception,dash,report]"

if [ -z "${VRGRID_NO_GPU:-}" ]; then
  echo "==> cupy (with the CUDA toolkit headers it JITs against)"
  .venv/bin/pip -q install "cupy-cuda12x[ctk]"

  # cupy ships the toolkit as a pip package but does not always find it, and
  # the failure is at FIRST KERNEL LAUNCH rather than at import -- so it looks
  # like a broken GPU rather than a missing header. This .pth resolves it
  # relative to the venv, so it is correct on any machine without anyone
  # exporting CUDA_PATH by hand.
  SITE="$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
  cat > "$SITE/_vrgrid_cuda_path.pth" <<'PTH'
import os, os.path as _p, site as _s; _c = _p.join(_s.getsitepackages()[0], 'nvidia', 'cuda_runtime'); os.environ.setdefault('CUDA_PATH', _c) if _p.isdir(_p.join(_c, 'include')) else None
PTH

  echo "==> torch (cu128; sm_120 / Blackwell needs 12.8 or newer)"
  .venv/bin/pip -q install torch --index-url https://download.pytorch.org/whl/cu128
fi

echo
echo "==> verifying"
.venv/bin/python scripts/verify_env.py
