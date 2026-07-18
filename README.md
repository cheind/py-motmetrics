[![PyPI version](https://badge.fury.io/py/motmetrics.svg)](https://badge.fury.io/py/motmetrics) [![Build Status](https://github.com/cheind/py-motmetrics/actions/workflows/python-package.yml/badge.svg)](https://github.com/cheind/py-motmetrics/actions/workflows/python-package.yml) [![DOI](https://zenodo.org/badge/87559569.svg)](https://doi.org/10.5281/zenodo.14014773)

# py-motmetrics

**py-motmetrics** provides Python tools for evaluating multiple object tracking (MOT) results. It implements MOTChallenge-aligned CLEAR MOT, Identity, and HOTA-related metrics, including MOTA, MOTP, IDF1, precision, recall, and track quality counts.

## Installation

```bash
pip install motmetrics
```

Python 3.8 through 3.14 is supported.

To materialize results as a pandas dataframe through `summary.df`, install the
optional dataframe extra:

```bash
pip install "motmetrics[dataframe]"
```

For development:

```bash
uv venv
uv pip install --group dev
```

## Quick Start

For MOTChallenge-style text files, compute and print metrics in one call. Supported file formats are detected automatically.

```python
import motmetrics as mm

summary = mm.evaluate_motchallenge("path/to/gt.txt", "path/to/pred.txt")
print(summary)
```

`summary` displays as a MOTChallenge-style table. Read scalar values or complete
metric columns without pandas:

```python
hota_by_sequence = summary["hota"]
sequence_hota = summary[summary.index[0], "hota"]
```

Column access returns an ordered mapping from sequence name to value. Folder
evaluation also provides the combined value as `summary["OVERALL", "hota"]`.

With the dataframe extra installed, a pandas view remains available on demand:

```python
summary.df["mota"]
summary.df["idf1"]
summary.df["hota"]
summary.df.to_csv("metrics.csv")
```

Evaluation, native metric access, and text rendering do not import pandas. The
dataframe is materialized only when `.df` is accessed.

By default, `evaluate_motchallenge` uses `fmt="auto"`. It detects MOTChallenge text, VATIC text, and UA-DETRAC XML files. For ambiguous text files, pass the format explicitly:

```python
summary = mm.evaluate_motchallenge(gt, pred, fmt="mot16")
```

Folder evaluation uses the same function:

```python
summary = mm.evaluate_motchallenge(
    "path/to/gt_root",
    "path/to/preds_root",
    n_jobs=4,
)
print(summary)
```

For folder-based IoU evaluation, `n_jobs > 1` runs complete sequences in
separate processes, including file loading, CLEAR/Identity metrics, and HOTA.
Use at most the number of sequences and benchmark against the number of
physical CPU cores; extra processes can be slower once process overhead or
memory bandwidth becomes the bottleneck.

Interactive terminals automatically show one progress row per sequence. Pass
`progress=False` to suppress it, or `progress=True` to force it when stderr is
not detected as an interactive terminal.

Expected folder layout:

```text
gt_root/<SEQUENCE>/gt/gt.txt
preds_root/<SEQUENCE>.txt
```

## Metrics

The MOTChallenge summary always includes the commonly reported CLEAR, Identity,
and HOTA metrics. HOTA diagnostics include DetRe, DetPr, AssRe, AssPr, LocA,
and OWTA. CLEAR diagnostics include MODA, sMOTA, MTR, PTR, MLR, CLR_F1, and
false positives per frame.

### Metric reference

The display name is the column printed in the summary. The Python name is used
by `summary[...]`, `summary.df`, and the optional `metrics=` argument. Ratios
and accuracies are stored in the range `[0, 1]` before display formatting,
although error-based scores such as MOTA, MODA, and sMOTA can be negative.

#### Identity and detection

| Display | Python name | Meaning | Better |
|---|---|---|:---:|
| IDF1 | `idf1` | F1 score of correctly identified detections under the global trajectory assignment. | Higher |
| IDP | `idp` | Fraction of predicted detections whose identity is correct. | Higher |
| IDR | `idr` | Fraction of ground-truth detections whose identity is recovered correctly. | Higher |
| Rcll | `recall` | CLEAR detection recall: matched detections divided by ground-truth detections. | Higher |
| Prcn | `precision` | CLEAR detection precision: matched detections divided by all tracker detections. | Higher |
| FP | `num_false_positives` | Tracker detections that were not matched to ground truth. | Lower |
| FN | `num_misses` | Ground-truth detections that the tracker missed. | Lower |
| IDs | `num_switches` | Times a ground-truth identity changes from its previously assigned tracker identity. | Lower |
| FM | `num_fragmentations` | Interruptions where a tracked ground-truth trajectory becomes missed and is later reacquired. | Lower |

IDP, IDR, and IDF1 use a global one-to-one trajectory assignment, rather than
the frame-local assignments used by CLEAR metrics:

```text
IDP  = IDTP / (IDTP + IDFP)
IDR  = IDTP / (IDTP + IDFN)
IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
```

#### Track coverage

| Display | Python name | Meaning | Better |
|---|---|---|:---:|
| GT | `num_unique_objects` | Number of unique ground-truth trajectories. | Context |
| MT | `mostly_tracked` | Ground-truth trajectories matched for more than 80% of their lifespan. | Higher |
| PT | `partially_tracked` | Ground-truth trajectories matched for 20% through 80% of their lifespan. | Context |
| ML | `mostly_lost` | Ground-truth trajectories matched for less than 20% of their lifespan. | Lower |
| MTR | `mtr` | Mostly-tracked ratio: `MT / GT`. | Higher |
| PTR | `ptr` | Partially-tracked ratio: `PT / GT`. | Context |
| MLR | `mlr` | Mostly-lost ratio: `ML / GT`. | Lower |

#### CLEAR scores

| Display | Python name | Meaning | Better |
|---|---|---|:---:|
| MOTA | `mota` | Tracking accuracy penalizing false negatives, false positives, and identity switches. | Higher |
| MODA | `moda` | Detection accuracy penalizing false negatives and false positives, but not identity switches. | Higher |
| MOTP | `motp` | Mean localization distance over CLEAR matches. Identical boxes have distance zero. | Lower |
| sMOTA | `smota` | Soft MOTA, which also rewards the localization similarity of matched detections. | Higher |
| CLR_F1 | `clr_f1` | Harmonic mean of CLEAR detection precision and recall. | Higher |
| FP/Frame | `fp_per_frame` | Mean number of false-positive detections per evaluated frame. | Lower |

Using `TP` for CLEAR matches, `G = TP + FN` for the number of ground-truth
detections, `S` for the sum of matched localization similarities, and `F` for
the number of evaluated frames:

```text
Recall   = TP / G
Precision = TP / (TP + FP)
MOTA     = 1 - (FN + FP + IDs) / G
MODA     = (TP - FP) / G
sMOTA    = (S - FP - IDs) / G
CLR_F1   = 2 * TP / (2 * TP + FN + FP)
FP/Frame = FP / F
```

The displayed `GT` column counts unique trajectories. It is not `G`, the
ground-truth detection count used as the denominator of MOTA, MODA, and sMOTA.

#### Identity event diagnostics

| Display | Python name | Meaning | Better |
|---|---|---|:---:|
| IDt | `num_transfer` | A tracker identity transfers from its previously assigned ground-truth identity to another one. | Lower |
| IDa | `num_ascend` | A ground-truth identity switches to a tracker identity that has not been matched before. | Lower |
| IDm | `num_migrate` | A tracker identity transfers to a ground-truth identity that has not been matched before. | Lower |

These event diagnostics describe how an identity error happened. `IDs` remains
the standard identity-switch count used by MOTA.

#### HOTA scores

| Display | Python name | Meaning | Better |
|---|---|---|:---:|
| HOTA | `hota` | Geometric mean of detection accuracy and association accuracy. | Higher |
| DetA | `deta` | Jaccard detection accuracy over HOTA matches, misses, and false positives. | Higher |
| AssA | `assa` | Association Jaccard accuracy, averaged over matched detections. | Higher |
| DetRe | `detre` | Detection recall at the HOTA matching thresholds. | Higher |
| DetPr | `detpr` | Detection precision at the HOTA matching thresholds. | Higher |
| AssRe | `assre` | Fraction of each matched ground-truth trajectory association that is recovered. | Higher |
| AssPr | `asspr` | Fraction of each matched predicted trajectory association that is correct. | Higher |
| LocA | `loca` | Mean localization similarity of HOTA true-positive matches. | Higher |
| OWTA | `owta` | Open-world tracking accuracy, balancing detection recall and association accuracy. | Higher |

For every HOTA localization threshold `alpha`:

```text
DetRe = HOTA_TP / (HOTA_TP + HOTA_FN)
DetPr = HOTA_TP / (HOTA_TP + HOTA_FP)
DetA  = HOTA_TP / (HOTA_TP + HOTA_FN + HOTA_FP)
HOTA  = sqrt(DetA * AssA)
OWTA  = sqrt(DetRe * AssA)
```

The displayed HOTA-family values are means over the configured thresholds. By
default these are `0.05, 0.10, ..., 0.95`, matching TrackEval. `LocA` uses
similarity, while this package's CLEAR `MOTP` column uses distance.

`motmetrics.evaluate_motchallenge` is the only supported metrics entrypoint.
The accumulator, matching, dependency resolution, and per-sequence process
workers are internal implementation details so every invocation follows the
same optimized computational path.

For the full HOTA/CLEAR/Identity parity check against TrackEval, see [motmetrics/tests/test_trackeval_parity.py](motmetrics/tests/test_trackeval_parity.py).

## MOTChallenge Notes

Results are aligned with the MOTChallenge devkit, with one format difference:

- MOTChallenge reports MOTP as a percentage, while py-motmetrics reports the average distance. Convert with `(1 - MOTP) * 100` for MOTChallenge-style MOTP.

## Development

Run the test suite:

```bash
uv run --no-project pytest
```

Run the TrackEval parity test locally:

```bash
uv pip install trackeval==1.3.0
uv run --no-project pytest -q motmetrics/tests/test_trackeval_parity.py
```

## References

1. Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics." EURASIP Journal on Image and Video Processing, 2008.
2. Milan, Anton, et al. "MOT16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831, 2016.
3. Li, Yuan, Chang Huang, and Ram Nevatia. "Learning to associate: HybridBoosted multi-target tracker for crowded scene." CVPR, 2009.
4. Ristani, Ergys, et al. "Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking." ECCV Workshop, 2016.

## License

MIT. See [LICENSE](LICENSE).
