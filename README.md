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

`summary` displays as a MOTChallenge-style table. With the dataframe extra
installed, the raw pandas data is available on demand:

```python
summary.df["mota"]
summary.df["idf1"]
summary.df["hota"]
summary.df.to_csv("metrics.csv")
```

The evaluation and text rendering path does not import pandas. The dataframe is
materialized only when `.df` is accessed.

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

The MOTChallenge summary always includes the commonly reported CLEAR, Identity, and HOTA metrics.

`motmetrics.evaluate_motchallenge` is the only supported metrics entrypoint.
The accumulator, matching, dependency resolution, and per-sequence process
workers are internal implementation details so every invocation follows the
same optimized computational path.

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
