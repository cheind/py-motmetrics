[![PyPI version](https://badge.fury.io/py/motmetrics.svg)](https://badge.fury.io/py/motmetrics) [![Build Status](https://github.com/cheind/py-motmetrics/actions/workflows/python-package.yml/badge.svg)](https://github.com/cheind/py-motmetrics/actions/workflows/python-package.yml) [![DOI](https://zenodo.org/badge/87559569.svg)](https://doi.org/10.5281/zenodo.14014773)

# py-motmetrics

**py-motmetrics** provides Python tools for evaluating multiple object tracking (MOT) results. It implements MOTChallenge-aligned CLEAR MOT, Identity, and HOTA-related metrics, including MOTA, MOTP, IDF1, precision, recall, and track quality counts.

## Installation

```bash
pip install motmetrics
```

Python 3.8 through 3.14 is supported. NumPy, pandas, SciPy, and xmltodict are installed as dependencies.

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

`summary` displays as a MOTChallenge-style table and keeps the raw pandas data available:

```python
summary.mota
summary.idf1
summary.hota
summary.df.to_csv("metrics.csv")
```

By default, `evaluate_motchallenge` uses `fmt="auto"`. It detects MOTChallenge text, VATIC text, and UA-DETRAC `.mat`/`.xml` files. For ambiguous text files, pass the format explicitly:

```python
summary = mm.evaluate_motchallenge(gt, pred, fmt=mm.io.Format.MOT16)
```

Folder evaluation uses the same function:

```python
summary = mm.evaluate_motchallenge("path/to/gt_root", "path/to/preds_root")
print(summary)
```

Expected folder layout:

```text
gt_root/<SEQUENCE>/gt/gt.txt
preds_root/<SEQUENCE>.txt
```

The command-line evaluator is still available:

```bash
python -m motmetrics.apps.eval_motchallenge path/to/gt_root path/to/preds_root
```

## Metrics

List all registered metrics:

```python
import motmetrics as mm

print(mm.list_metrics_markdown())
```

The default MOTChallenge summary includes the commonly reported CLEAR, Identity, and HOTA metrics.

## Advanced Use

Useful lower-level pieces:

- `mm.MOTAccumulator` stores frame-level matching events.
- `mm.distances` contains distance helpers such as IoU and Euclidean matrices.
- `mm.io.loadtxt(..., fmt="auto")` detects MOTChallenge text, VATIC text, and UA-DETRAC MAT/XML files.
- `mm.metrics.create()` returns a `MetricsHost` for custom metric selection.
- `mm.utils.compare_to_groundtruth` compares loaded dataframes directly.
- `mm.utils.compare_to_groundtruth_reweighting` supports custom HOTA-style multi-threshold workflows.

For the full HOTA/CLEAR/Identity parity check against TrackEval, see [motmetrics/tests/test_trackeval_parity.py](motmetrics/tests/test_trackeval_parity.py).

## MOTChallenge Notes

Results are aligned with the MOTChallenge devkit, with two naming/format differences:

- `FAR` is not listed directly; it can be computed as false positives per frame.
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