"""Time ``motmetrics.evaluate_motchallenge`` on arbitrary evaluation inputs.

Example:

    python examples/benchmark_motmetrics.py path/to/gt path/to/predictions --jobs 4
"""

import argparse
import statistics
import time

import motmetrics as mm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", help="Ground-truth file or evaluation root")
    parser.add_argument("predictions", help="Prediction file or evaluation root")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Sequence worker processes (default: max(1, CPU count - 2) for folder evaluation)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of measured evaluations (default: 1)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Optional metric names to calculate",
    )
    args = parser.parse_args()
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    from pathlib import Path
    matched_files = mm._evaluation._match_input_files(Path(args.ground_truth), Path(args.predictions))
    actual_jobs = mm._evaluation._validate_n_jobs(args.jobs, num_tasks=len(matched_files))

    timings = []
    summary = None
    for _ in range(args.runs):
        started = time.perf_counter()
        summary = mm.evaluate_motchallenge(
            args.ground_truth,
            args.predictions,
            metrics=args.metrics,
            n_jobs=args.jobs,
            progress=False,
        )
        timings.append(time.perf_counter() - started)

    print(summary)
    if args.runs == 1:
        print(
            "\nevaluate_motchallenge: {:.3f}s (n_jobs={})".format(
                timings[0],
                actual_jobs,
            )
        )
        return

    print(
        "\nevaluate_motchallenge: median {:.3f}s, min {:.3f}s, max {:.3f}s "
        "({} runs, n_jobs={})".format(
            statistics.median(timings),
            min(timings),
            max(timings),
            args.runs,
            actual_jobs,
        )
    )


if __name__ == "__main__":
    main()
