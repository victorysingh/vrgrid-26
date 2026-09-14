# PROVENANCE -- written 2026-09-14 for D11 Step 5. Diagnostic only; changes nothing.
# Run: python reports/harnesses/shim_determinism_probe.py
"""Is --fast-scatter deterministic on CPU at the REAL shapes? Read-only diagnostic, no fix.

Step 2's verify is bit-identical at 20,000 x 64 into 4,000 slots. Step 5 showed per-class IoU
varying between two --fast-scatter runs while the loop arm was identical run to run. This checks
the reductions alone at the benchmark shape (124,000 x 256 into 25,000), repeated, at the default
thread count and at 1 thread, against the port's own loop.
"""
import argparse
import hashlib
import sys
import torch
sys.path.insert(0, "scripts")
from vrgrid.perception.frnet import frustum_encoder as fe  # loop versions, before any patch  # noqa: E402
import frnet_fast_scatter as fs  # noqa: E402

loop_max, loop_mean = fe.scatter_max, fe.scatter_mean
_ap = argparse.ArgumentParser()
_ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
_args = _ap.parse_args()
DEV = torch.device(_args.device)
if DEV.type == "cuda" and not torch.cuda.is_available():
    sys.exit("--device cuda requested but CUDA is not available")

N, C, SLOTS, REPS = 124_000, 256, 25_000, 6
g = torch.Generator().manual_seed(0)
src = torch.randn(N, C, generator=g).to(DEV)
index = torch.randint(0, SLOTS, (N,), generator=g).to(DEV)

def run(fn, threads):
    torch.set_num_threads(threads)
    outs = []
    for _ in range(REPS):
        o = fn(src, index, dim=0, dim_size=SLOTS)
        outs.append((o[0] if isinstance(o, tuple) else o).clone())
    return outs

default_threads = torch.get_num_threads()
print(f"torch {torch.__version__}, device {DEV}, default threads {default_threads}, "
      f"shape {N}x{C} into {SLOTS}, {REPS} repeats")
for name, fast, loop in (("scatter_max", fs.fast_scatter_max, loop_max),
                         ("scatter_mean", fs.fast_scatter_mean, loop_mean)):
    torch.set_num_threads(default_threads)
    ref = loop(src, index, dim=0, dim_size=SLOTS)
    ref = ref[0] if isinstance(ref, tuple) else ref
    for threads in (default_threads, 1):
        outs = run(fast, threads)
        repeats_identical = all(torch.equal(outs[0], o) for o in outs[1:])
        n_diff_between = sum(int((outs[0] != o).sum()) for o in outs[1:])
        vs_loop = [float((o - ref).abs().max()) for o in outs]
        eq_loop = sum(torch.equal(o, ref) for o in outs)
        digest = hashlib.sha256(outs[0].detach().cpu().numpy().tobytes()).hexdigest()[:16]
        print(f"  {name:<13} threads={threads:<2} repeats bit-identical: {repeats_identical!s:<5} "
              f"sha256[0]={digest} "
              f"(values differing across repeats: {n_diff_between:,})  "
              f"== loop in {eq_loop}/{REPS}  max |fast-loop| {max(vs_loop):.3e}")
torch.set_num_threads(default_threads)
