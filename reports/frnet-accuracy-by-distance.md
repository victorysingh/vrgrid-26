# FRNet classification accuracy across distance bands (SIH26053 "Performance Metrics")

**Asked for by the problem statement:** *"evidence of … high accuracy in object classification
across varying distances."* This report is that evidence, with its limits stated first.

**Model:** the independently sourced public FRNet checkpoint, SHA-256
`09adea9005215641aea915cc3aa2bebf74582ce240cca91dedd07940ad94285e`. It is **not** the 4 Sep file; see
`checkpoints/frnet-semantickitti_seg.pth.PROVENANCE.md`.
**Run:** `scripts/frnet_eval_by_range.py --seq 08 --frames 200 --fast-scatter --threads 1`, on a
clean machine: clock 2400/2400 MHz, commit 12.83 → 12.66 of 15.73 GB, state OK before and after,
checkpoint SHA-256 verified. That configuration is reproducible per point (OPEN-ITEMS R-j).
Evidence: `reports/bench/frnet_eval_by_range_seq08.json`.

---

## Limits, first

1. **No accuracy can be measured beyond 50 m on this dataset.** In SemanticKITTI seq 08 every point
   beyond 50.0 m horizontal range is labelled `unlabeled` (raw id 0). The map's grid extends to
   ±100 m, and the problem statement's example goes to 100 m; that outer half of the range has no
   ground truth to score against (see "The 50 m label ceiling" below).
2. **mIoU is not comparable across bands.** Each band's mIoU averages only the classes present
   there: 11 classes at 0–10 m, 15 beyond. Compare the group IoUs and the point accuracy across
   bands instead.
3. **The bands are radial and the map's rings are square windows** (±10 / ±25 / ±50 / ±100 m). Near a
   ring corner a point can fall in the next band out, so band ↔ ring is an approximate
   correspondence.
4. **One sequence (08, the standard validation sequence), first 200 frames, one checkpoint.**
   "Movable object" is a class property; FRNet has no motion output, so this does **not** measure
   moving-object detection.

## Built-in check: the bands reproduce the known whole-slice numbers

- **Summed over all bands: point accuracy 90.3036%, mIoU 65.18% over 15 classes.** Tonight's
  reproducible single-thread value is 90.30359%, and `frnet_eval.py` records 65.2%, so both agree.
- **Band point counts sum to 22,741,893,** the whole labelled slice: every labelled point was counted
  exactly once.

## Results by distance band

| band | labelled points | share | point accuracy | mIoU (classes present) | **drivable terrain IoU** | **static obstacle IoU** | **movable object IoU** | 3-group accuracy |
|---|---|---|---|---|---|---|---|---|
| **0–10 m** | 11,468,750 | 50.4% | **93.20%** | 65.4% (11) | 94.1% | 67.2% | 98.9% | 95.10% |
| **10–25 m** | 8,220,174 | 36.1% | **88.10%** | 67.0% (15) | 86.1% | 82.3% | 92.9% | 91.73% |
| **25–50 m** | 3,052,969 | 13.4% | **85.36%** | 55.8% (15) | 78.1% | 88.9% | 82.0% | 92.04% |
| 50–100 m | 0 | — | *no ground truth* | — | — | — | — | — |
| >100 m | 0 | — | *no ground truth* | — | — | — | — | — |
| **all** | 22,741,893 | 100% | **90.30%** | 65.2% (15) | 90.5% | 80.8% | 96.5% | 93.47% |

Groups follow the problem statement's wording, and both prediction and ground truth are collapsed to
the group (a car predicted as truck still counts as a correct "movable object"):

- **drivable terrain:** road, parking, sidewalk, other-ground, terrain (the map's §7.1 set);
- **static obstacle:** building, fence, vegetation, trunk, pole, traffic-sign;
- **movable object:** car, bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist.

### What the table shows

- **Accuracy degrades with distance, as expected for a lidar segmenter:** 93.2% → 88.1% → 85.4% point
  accuracy.
- **The classes that matter most for safety are strongest where the map is finest.** Within 10 m (the
  5 cm ring), movable objects score IoU 98.9% and drivable terrain 94.1%. They fall to 82.0% and 78.1%
  at 25–50 m.
- **Static obstacles go the other way,** 67.2% → 88.9%. Near the sensor they are dominated by small,
  hard classes (fence 33%, traffic-sign 12%, pole 59%); far away, by large, easy buildings (85–91%).
  So this is a class-mix effect, not better long-range perception.
- **3-group accuracy stays at 92–95% in every labelled band.**

## Per-class IoU by band

| class | 0–10 m | 10–25 m | 25–50 m |
|---|---|---|---|
| car | 99.2% | 96.1% | 86.1% |
| bicycle | — | 53.5% | 26.2% |
| person | — | 73.3% | 36.6% |
| bicyclist | 92.0% | 91.3% | 83.9% |
| road | 98.4% | 96.1% | 86.2% |
| parking | — | 49.4% | 43.1% |
| sidewalk | 92.6% | 87.1% | 64.0% |
| other-ground | 0.0% | 0.0% | 0.0% |
| building | — | 90.8% | 85.1% |
| fence | 33.1% | 37.8% | 22.1% |
| vegetation | 61.2% | 69.5% | 81.1% |
| trunk | 93.0% | 78.9% | 58.6% |
| terrain | 79.3% | 69.8% | 62.2% |
| pole | 59.0% | 52.0% | 49.9% |
| traffic-sign | 12.1% | 59.3% | 52.4% |

"—" means no ground truth of that class in that band. `other-ground` has 150 points in the whole slice,
all missed (as recorded on 4 Sep). The largest distance drops are for the small vulnerable classes:
person 73.3% → 36.6% and bicycle 53.5% → 26.2% from 10–25 m to 25–50 m.

## The 50 m label ceiling: measured, not assumed

The first result showed exactly zero labelled points beyond 50 m, so I checked the data directly
before reporting it (`reports/harnesses/semantickitti_label_range.py`, 11 frames sampled across the
slice):

| band | raw points | labelled (not `unlabeled`) | labelled % |
|---|---|---|---|
| 0–10 m | 649,114 | 634,204 | 97.70% |
| 10–25 m | 474,444 | 461,653 | 97.30% |
| 25–50 m | 171,422 | 166,000 | 96.84% |
| **50–100 m** | **60,942** | **0** | **0.00%** |
| >100 m | 0 | 0 | — |

- Farthest raw point: **80.5 m**. Farthest labelled point: **50.0 m**.
- Every point beyond 50 m carries raw semantic id **0** (`unlabeled`): 60,942 of 60,942.

So the zero is in the ground truth, not in the band assignment: SemanticKITTI seq 08 is not annotated
beyond 50 m horizontal range. FRNet still predicts classes out there, and the map still fuses them,
but no ground truth can score them.

## A defect caught before this run

The script's first unit-test run failed on its IoU tally. The confusion-count index used a row stride
of `n` over `n + 1` columns, so an out-of-range prediction (FRNet's ignore slot 19) spilled into the next
class's row, and a `-1` group prediction was clipped into group 0. Both would have corrupted per-band
IoUs on real data. It was fixed before any real run: stride `n + 1`, with every out-of-range prediction
routed to the extra column. `tests/test_frnet_eval_by_range.py` pins the tally against `frnet_eval.py`'s
own loop definition.
