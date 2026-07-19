<p align="center">
  <img src=".github/assets/logo.png" alt="py-motmetrics logo" width="500">
</p>

<p align="center"><strong>Fast, extensible multi-object tracking evaluation.</strong></p>

<p align="center">
  <a href="https://badge.fury.io/py/motmetrics"><img src="https://badge.fury.io/py/motmetrics.svg" alt="PyPI version"></a>
  <a href="https://github.com/cheind/py-motmetrics/actions/workflows/python-package.yml"><img src="https://github.com/cheind/py-motmetrics/actions/workflows/python-package.yml/badge.svg" alt="Build status"></a>
  <a href="https://github.com/cheind/py-motmetrics/actions/workflows/python-package.yml"><img src="https://img.shields.io/badge/TrackEval%201.3.0-parity-brightgreen" alt="TrackEval parity"></a>
  <a href="https://doi.org/10.5281/zenodo.14014773"><img src="https://zenodo.org/badge/87559569.svg" alt="DOI"></a>
</p>

## Why MOTMetrics

- **Fast:** 2.65–4.90x faster than TrackEval 1.3.0 in measured end-to-end benchmarks.
- **Complete:** CLEAR, Identity, and HOTA metrics with TrackEval parity.
- **Simple:** one evaluation API, sequence parallelism, and two runtime dependencies.
- **Extensible:** custom metrics can reuse shared statistics or define their own matching.

## Installation

```bash
pip install motmetrics
```

Python 3.8 through 3.14 is supported.

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

Expected folder layout:

```text
gt_root/<SEQUENCE>/gt/gt.txt
preds_root/<SEQUENCE>.txt
```

## Metrics

`evaluate_motchallenge` returns all built-in CLEAR, Identity, and HOTA metrics
by default. Use `metrics=` to select and order only the metrics you need:

```python
summary = mm.evaluate_motchallenge(
    gt,
    predictions,
    metrics=["mota", "idf1", "hota", "assa"],
)
```

Expand a group below for metric definitions.

<details>
<summary><strong>Identity and detection</strong></summary>

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

</details>

<details>
<summary><strong>Track coverage</strong></summary>

| Display | Python name | Meaning | Better |
|---|---|---|:---:|
| GT | `num_unique_objects` | Number of unique ground-truth trajectories. | Context |
| MT | `mostly_tracked` | Ground-truth trajectories matched for more than 80% of their lifespan. | Higher |
| PT | `partially_tracked` | Ground-truth trajectories matched for 20% through 80% of their lifespan. | Context |
| ML | `mostly_lost` | Ground-truth trajectories matched for less than 20% of their lifespan. | Lower |
| MTR | `mtr` | Mostly-tracked ratio: `MT / GT`. | Higher |
| PTR | `ptr` | Partially-tracked ratio: `PT / GT`. | Context |
| MLR | `mlr` | Mostly-lost ratio: `ML / GT`. | Lower |

</details>

<details>
<summary><strong>CLEAR scores</strong></summary>

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

</details>

<details>
<summary><strong>Identity event diagnostics</strong></summary>

| Display | Python name | Meaning | Better |
|---|---|---|:---:|
| IDt | `num_transfer` | A tracker identity transfers from its previously assigned ground-truth identity to another one. | Lower |
| IDa | `num_ascend` | A ground-truth identity switches to a tracker identity that has not been matched before. | Lower |
| IDm | `num_migrate` | A tracker identity transfers to a ground-truth identity that has not been matched before. | Lower |

These event diagnostics describe how an identity error happened. `IDs` remains
the standard identity-switch count used by MOTA.

</details>

<details>
<summary><strong>HOTA scores</strong></summary>

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

</details>

### Custom metric families

Extend `evaluate_motchallenge` with `extra_metric_families=`. See examples using
[shared statistics](examples/custom_metrics/from_shared_statistics.py) or
[custom matching](examples/custom_metrics/with_custom_matching.py).

## Performance

End-to-end median runtime on an Apple M3 Max across seven fresh runs per
setting, including imports, file loading, IoU, CLEAR, Identity, and HOTA. One
sequence worker is used per requested core, capped by the sequence count:

<table align="center">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Backend</th>
      <th align="right">1 core</th>
      <th align="right">2 cores</th>
      <th align="right">4 cores</th>
      <th align="right">8 cores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TUD (2 sequences)</td>
      <td>py-motmetrics</td>
      <td align="right">0.117 s</td>
      <td align="right">0.146 s</td>
      <td align="right">0.164 s</td>
      <td align="right">0.151 s</td>
    </tr>
    <tr>
      <td>TUD (2 sequences)</td>
      <td>TrackEval 1.3.0</td>
      <td align="right">0.573 s</td>
      <td align="right">0.492 s</td>
      <td align="right">0.498 s</td>
      <td align="right">0.495 s</td>
    </tr>
    <tr>
      <td>TUD (2 sequences)</td>
      <td>Speedup</td>
      <td align="right"><strong>4.90x</strong></td>
      <td align="right"><strong>3.37x</strong></td>
      <td align="right"><strong>3.04x</strong></td>
      <td align="right"><strong>3.28x</strong></td>
    </tr>
    <tr>
      <td>MOT17 (7 sequences)</td>
      <td>py-motmetrics</td>
      <td align="right">0.374 s</td>
      <td align="right">0.284 s</td>
      <td align="right">0.248 s</td>
      <td align="right">0.263 s</td>
    </tr>
    <tr>
      <td>MOT17 (7 sequences)</td>
      <td>TrackEval 1.3.0</td>
      <td align="right">1.006 s</td>
      <td align="right">0.753 s</td>
      <td align="right">0.679 s</td>
      <td align="right">0.696 s</td>
    </tr>
    <tr>
      <td>MOT17 (7 sequences)</td>
      <td>Speedup</td>
      <td align="right"><strong>2.69x</strong></td>
      <td align="right"><strong>2.65x</strong></td>
      <td align="right"><strong>2.74x</strong></td>
      <td align="right"><strong>2.65x</strong></td>
    </tr>
  </tbody>
</table>

## References

1. Luiten, Jonathon, et al. ["HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking."](https://doi.org/10.1007/s11263-020-01375-2) International Journal of Computer Vision, 2021.
2. Ristani, Ergys, et al. ["Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking."](https://doi.org/10.1007/978-3-319-48881-3_2) ECCV Workshop, 2016.
3. Milan, Anton, et al. ["MOT16: A Benchmark for Multi-Object Tracking."](https://arxiv.org/abs/1603.00831) arXiv:1603.00831, 2016.
4. Li, Yuan, Chang Huang, and Ram Nevatia. ["Learning to Associate: HybridBoosted Multi-Target Tracker for Crowded Scene."](https://doi.org/10.1109/CVPR.2009.5206735) CVPR, 2009.
5. Bernardin, Keni, and Rainer Stiefelhagen. ["Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics."](https://doi.org/10.1155/2008/246309) EURASIP Journal on Image and Video Processing, 2008.
