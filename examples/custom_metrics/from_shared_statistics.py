"""Compute track coverage (TCOV) as an opt-in metric family.

For each ground-truth trajectory, TCOV measures the fraction of its annotated
lifespan covered by any tracker identity linked to it through CLEAR matching.
The reported score is the mean coverage across ground-truth trajectories, so a
TCOV of 0.8 means that the tracker sees an average object for approximately 80%
of its lifetime in the evaluated frames.

Run this example with either two MOTChallenge files or two evaluation roots:

    python examples/custom_metrics/from_shared_statistics.py path/to/gt path/to/predictions
"""

import argparse

import motmetrics as mm


class TrackCoverage(mm.MetricFamily):
    """Average per-GT-trajectory lifespan coverage by linked tracker tracks."""

    name = "track_coverage"
    metric_names = ("tcov",)
    requirements = frozenset(("clear_statistics",))
    display_names = {"tcov": "TCOV"}
    formatters = {"tcov": "{:.1%}".format}

    def evaluate_sequence(self, sequence, intermediates):
        del sequence
        per_track_coverage = intermediates.clear_statistics.track_coverage

        # Preserve additive state so OVERALL can average trajectories rather
        # than incorrectly averaging already-normalized sequence scores.
        return float(per_track_coverage.sum()), len(per_track_coverage)

    def summarize(self, partial):
        coverage_sum, track_count = partial
        tcov = coverage_sum / max(1, track_count)
        if not 0.0 <= tcov <= 1.0:
            raise ValueError("TCOV must be between 0.0 and 1.0, got {!r}".format(tcov))
        return {"tcov": tcov}

    def combine(self, partials):
        coverage_sum = sum(partial[0] for partial in partials)
        track_count = sum(partial[1] for partial in partials)
        return self.summarize((coverage_sum, track_count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", help="Ground-truth file or evaluation root")
    parser.add_argument("predictions", help="Prediction file or evaluation root")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Sequence worker processes (default: 1)",
    )
    args = parser.parse_args()

    summary = mm.evaluate_motchallenge(
        args.ground_truth,
        args.predictions,
        n_jobs=args.jobs,
        extra_metric_families=TrackCoverage(),
        progress=False,
    )
    print(summary)


if __name__ == "__main__":
    main()
