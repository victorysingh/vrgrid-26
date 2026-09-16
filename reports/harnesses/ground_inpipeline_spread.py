# PROVENANCE -- written 2026-09-16 for OPEN-ITEMS R-h (the `ground` half) and
#              reports/r-b-p99-tail-investigation.md section 13.
#
# Produced: the in-pipeline `ground` p99-p50 spread per state-trusted run, before and after the
#           p99 allocation fixes, read from the committed whole-frame bench JSONs.
# Run:      python reports/harnesses/ground_inpipeline_spread.py
#
# NOTE: No new timing. Every whole-frame bench on 2026-09-14 already recorded a per-stage `ground`
#      row (p50 / p99 / max) per fresh-process run, with machine state checked before and after.
#      So the section-12 hypothesis -- "ground's 11.13 ms in-pipeline spread is other stages'
#      allocation, and collapses once the transform/cleanup fixes are in" -- can be tested on
#      data that already exists, measured under D8 discipline. Only runs whose state was trusted
#      before AND after are counted.
#
# Measurement only (reads JSON). Changes nothing.
"""In-pipeline `ground` spread across the committed trusted benches, before vs after the fixes."""
import json
import statistics as st
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "bench"
STATES = [
    ("baseline", "before any fix (main@2c952dd)"),
    ("after_transform", "+ transform scratch (12613df)"),
    ("after_both", "+ cleanup LUT (d540618)"),
    ("after_reflectivity_centres", "+ reflectivity skip, _centres (35b7b27, 697a2bd)"),
    ("after_range_image", "+ range_image selection (b9eef67)"),
    ("pooled_after_range_image", "same code, pooled-gate bench (36a4dbe)"),
]
BEFORE = ("baseline",)
AFTER = ("after_range_image", "pooled_after_range_image")
EARLIER_FIGURE_MS = 11.13   # warm p99 probe, 2026-09-13, machine state not recorded


def main():
    spreads = {}
    print(f"{'bench':<28}{'runs':>5}{'trusted':>8}  {'ground p99-p50 per trusted run (ms)':<40}"
          f"{'median':>7}{'worst frame':>12}")
    for name, desc in STATES:
        d = json.loads((BENCH / f"{name}.json").read_text(encoding="utf-8"))
        tr = [r for r in d["reps"] if r.get("trusted") and "ground" in r["rows"]]
        sp = [round(r["rows"]["ground"]["p99"] - r["rows"]["ground"]["p50"], 2) for r in tr]
        worst = max(r["rows"]["ground"]["max"] for r in tr)
        spreads[name] = sp
        print(f"{name:<28}{len(d['reps']):>5}{len(tr):>8}  {str(sp):<40}{st.median(sp):>7.2f}{worst:>12.2f}"
              f"   {desc}")
    pre = [x for n in BEFORE for x in spreads[n]]
    post = [x for n in AFTER for x in spreads[n]]
    print(f"\nBEFORE fixes: {len(pre)} trusted runs, median {st.median(pre):.2f} ms, "
          f"range {min(pre):.2f}-{max(pre):.2f}")
    print(f"AFTER  fixes: {len(post)} trusted runs, median {st.median(post):.2f} ms, "
          f"range {min(post):.2f}-{max(post):.2f}")
    every = [x for v in spreads.values() for x in v]
    print(f"largest spread in ANY trusted run: {max(every):.2f} ms; "
          f"earlier in-pipeline figure under test: {EARLIER_FIGURE_MS} ms")
    print("reproduced on a trusted run: "
          + ("YES" if max(every) >= 0.8 * EARLIER_FIGURE_MS else "NO"))


if __name__ == "__main__":
    main()
