# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Fast MOTChallenge evaluation."""

from motmetrics._evaluation import evaluate_motchallenge

__all__ = ["evaluate_motchallenge"]


def __getattr__(name):
    if name == "__version__":
        from importlib.metadata import version

        value = version("motmetrics")
        globals()[name] = value
        return value
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
