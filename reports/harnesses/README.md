# Measurement harnesses

The scripts that produced the numbers in `reports/` and `pending-review/`.

They live here, next to the reports they justify, because **twice in two days a
published figure turned out to be unverifiable for exactly one reason: its
harness was session scratch and was never committed.**

- **p50 80.78 / p99 97.72 ms** — no log, no artifact, no script, no method in the
  commit. It took a full investigation to establish that the number was probably
  sound and only its *label* was wrong.
- **Seq 00 ring-1 RMSE 6.77 cm** — differed from a re-measurement by a 61–89
  scored-cell population difference, and could not be closed while the harness
  was missing. It closed within minutes of the harness being recovered: the
  script runs all three sequences in one process, so seq 00's Patchwork++
  estimator carries ~160 frames of prior history. **That is a one-line fact that
  cost two investigations to rediscover.**

In both cases the figure itself was fine. The missing artifact was the defect.
So: **a harness that produces a number a report will quote is a deliverable,
whatever its filename says.**

---

## Conventions

Every file carries a `# PROVENANCE` header naming:

- the report it produced, and **which figures in it**
- the exact invocation, including `VRGRID_DATA_ROOT`
- any trap that cost time when the harness was written

They are **measurement only**. They read the shipping code and change nothing in
`src/`. Where a script needs a configuration the shipping code does not use — a
different Patchwork++ parameter set, say — it builds its own estimator rather
than editing `src/perception/ground.py`.

## Running them

All real-data harnesses need the dataset root set:

```sh
VRGRID_DATA_ROOT=C:/KITTI/dataset python reports/harnesses/<name>.py <args>
```

Without it, `loader.DATA_ROOT` falls back to `./data`, which is a **partial
stub** — calib and poses only, 49 scans, no labels — and the failure reads as
"the dataset is missing" rather than "the variable is unset". See
`src/perception/loader.py::_data_root_hint`.

Most take `<sequence> <frames>`; the R7 group takes a dumped `costmaps.npz`
(build it once with `dump_costmaps.py`, then iterate predicates instantly).

## [!] Two traps that have each cost real time

1. **`height_rmse_per_ring` returns CENTIMETRES**, and a `{ring: rmse}` dict.
   Multiplying by 100 gives a table exactly 100× out but otherwise plausible;
   iterating the return value yields ring *indices* (0, 1, 2, 3), which looks
   like a clean ascending RMSE curve and is not one. **Both mistakes produced
   believable tables here before a gate caught them** — which is why several of
   these scripts gate on a published number before printing anything. Keep the
   gates.

2. **The R7 planning window is not centred on the vehicle.** It is
   `x0 = vx - 11.0`, `y0 = vy - 5.5`, 44×44 — entirely *behind* it, over ground
   it has actually driven and therefore observed. Centring it drops seq 07
   support from 1,724 to 955 and the counts stop reproducing. R7's drivability
   predicate is also undocumented and was recovered by enumerating all 63 bit
   masks: it is `TRAV_SLOPE | TRAV_STEP` only — **not** roughness, **not** class.

## The gate habit

Several harnesses reproduce a known-good published number *before* reporting
anything derived. `numiter_accuracy.py` will not print a comparison until the
shipped configuration reproduces the published ring-0 RMSE; `numiter_r7.py`
reports its gate against the published support / non-drivable / miss triple.

This is not ceremony. It caught a 100× unit error, a wrong planning window, and
a wrong drivability predicate — each of which had already produced a plausible
table.

## What is deliberately not here

One-off scripts that *applied an edit* rather than measuring something
(`fix_miou.py`, `fix_latency_line.py`, `fix_determinism_claim.py`,
`patch_loader.py`, `resolve.py`) and plain `git show` copies of `timing_table.py`
at various refs, used once for a `--help` regression check. None of them produced
a figure any report quotes.
