# Morning summary 6 — pre-AWS pass (2026-09-16)

Covers what was done while the AWS credentials were being obtained: the pre-launch audit and its
fixes, the OPEN-ITEMS clean-up, the laptop-side transfer files, and two results that needed no AWS
access. Local branch `jp/p99-alloc-fixes`, **nothing pushed.**

---

## Still open, first

1. **The AWS GPU run** (DL pipeline real-time figure, GPU reproducibility): waiting on JP's
   credentials and instance launch. The laptop side is ready (below).
2. **The README / report pass** waits for the AWS numbers.
3. **Waiting on people, unchanged:** Shrestha (review of `d540618`, `697a2bd`); JP + Shrestha (D8);
   Aakash (D2, D3); the room / three-way (D3, D6, D10); JP (D1, D4, D5, D7, R-c, R-i, push / PR).
4. **ROS adapter (D10):** design and step-1 prep only, unchanged since 13 Sep. Step 2 (the
   vectorised bulk reader) is held pending the scope decision.
5. **CI lint rule ISC004** can't be checked locally (ruff 0.12.0 doesn't know it). RUF059, C408,
   UP030 and UP032 pass on all 28 changed Python files.

## Done in this pass

| what | result | commit |
|---|---|---|
| Pre-AWS status audit | Read-only. Found the runbook could not have worked as written (unpushed branch, one frame short, incomplete install, wrong Linux memory rule, no transfer, retrieval or termination steps). | — |
| Runbook fixed | Git-bundle transfer, frames 0–200 (494 MB), CUDA torch + `.[perception]`, a Patchwork++ check, an on-instance 2-frame preflight, Linux state rules, results fetched before termination, a termination checklist. Laptop-testable parts tested; tar needs `--force-local` on Windows. | `9b784e2` |
| OPEN-ITEMS clean-up | One current status per item; open items moved out of "Closed"; D9 closed; new rows for the DL mode, distance accuracy, runbook, push; stale `pending-review` headers corrected. | `d10cd9e` |
| Missing commit hashes | Zero-allocation scoping `235986d`; R-g `8b40e44`. | `66c4084` |
| Laptop transfer files | `~/vrgrid.bundle` (branch @ `d10cd9e`), `~/seq08_0-200.tar` (404 files), and `~/aws-transfer-manifest.txt` with SHA-256s; checkpoint re-verified. | not in git (by design) |
| DL mode re-checked live | `python -m vrgrid.run --seq 08 --frames 2 --semantics frnet --fast-scatter --threads 1`: exit 0, 413/413 parameters, Patchwork++ ground. | — |
| Full suite at HEAD | 728 passed / 1 failed (D1, known) / 3 skipped. | — |
| **R-h `ground`, answered without new timing** | In-pipeline `ground` p99−p50 on trusted runs: median **1.40 ms before any fix**, **2.46 ms after** (max 7.67). The earlier 11.13 ms never reproduces, so the allocation hypothesis is **not supported**. Machine state is the likely, but unprovable, cause. Rare 26–33 ms frames remain unattributed. | `e10db7a` |
| **D11 delta, paired per query** | Longitudinal **+1.069**, 95% CI **[+0.730, +1.436]**, 49/1/14 worse/equal/better; lateral **+0.567**, 95% CI **[+0.301, +0.846]**, 25/31/8. Both exclude zero; the oracle control is exactly zero; aggregates identical to the earlier reproducible run. The CI covers these 64 queries on one map, not other scenes. | this pass's final commit |

## Why no timing work ran

The machine was paging for most of this pass (commit 17–18 GB against 15.73 GB physical: a browser
open for AWS). Timing needs a clean machine, so both results above came from work where load doesn't
change the answer: re-analysing trusted committed data, and single-threaded runs that are
bit-identical regardless of load.
