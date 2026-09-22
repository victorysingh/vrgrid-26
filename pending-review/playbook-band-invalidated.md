# pending-review: `defense-rehearsal-playbook.md` carries pre-band-change figures — and it is panel-facing

**For Shrestha. Nothing edited in your file.** Raised separately from the general audit findings
because this is the highest-stakes of the three documents affected: it is what gets said to a panel.

## The issue, in one paragraph

`843ad54` (17 Sep) changed `vertical_extent_m` from `[-2.0, 6.0]` to `[-3.5, 4.5]` in both schedule
configs and wired it through `quantise.py`, `metrics.py`, `harness.py`, `kernels.py` and
`cuda_kernels.py`. **You propagated that into `docs/known-limitations.md` (137 lines) and four
presentation docs** — `00-START-HERE`, `02-WHAT-TO-PRESENT`, `05-PANEL-DEFENSE`, `07-CORRECTED-SCRIPT`.
`docs/defense-rehearsal-playbook.md` was **not** in that set. Its last commit is 2026-09-02, two weeks
before the band moved.

## The figures at risk in it

All of these derive from heights, so a 1.5 m shift in the band can move them:

- **"4,041 curb cells on 07 and 9,499 on 08"**, with ring medians quoted alongside — curb detection
  is a height-gradient test.
- **"drops 52.3% of sequence 07's peak occupied set"** — the occupied set depends on what is in band.
- The Eq. (22a) passage about differencing *"both geometric bits over a fixed physical baseline of
  0.50 m"*, and the kerb reading passable at every lattice.

**I have not re-measured any of them**, so this is "provisionally invalidated pending re-measure",
not "wrong". They may well survive unchanged — the band is the same 8 m width and only the floor and
ceiling moved — but a rehearsal document is the wrong place to find out.

## What is NOT affected

Checked and clear: README's memory ratios (21.5×, 286×) and cell counts, FRNet classification
accuracy, and all latency figures. Those are band-independent.

## Why this is separate from the process point

There is a second finding about `843ad54` — that `configs/` requires a three-way review per
`.github/CODEOWNERS` and the commit went in linear with no PR. **That is a process point and it is
not what this note is about**, because the change itself is well-evidenced: the split came from
surveying ground returns beyond 10 m on all eleven sequences, and you verified CPU/GPU bit-identical
on 200 frames of all eleven plus all 4,071 frames of seq 08. The technical work is sound. This note
is only about one document that the propagation missed.

## Suggested handling

Either re-run the curb and occupancy figures on the current band and update them in place, or add a
dated banner saying the numbers predate the 17 Sep rebalance — the same convention used elsewhere in
the project. **Which one is your call**; the deadline pressure on a rehearsal doc may make the banner
the right answer for now.

## For symmetry — the two equivalents on JP's side

The same sweep flagged two of JP's own reports as band-invalidated, and he owns fixing those:
`reports/r1-accuracy-by-class-and-range-band.md` (height RMSE per band × class) and
`reports/r7-hazard-miss-rate.md` (drivability miss rates). Both are marked in place. This is not a
finding about your lane specifically — it caught three documents and two of them are his.
