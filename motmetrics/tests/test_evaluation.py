import importlib.util
import subprocess
import sys
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import pytest
from pytest import approx

import motmetrics as mm
import motmetrics._evaluation as evaluation

DATA_DIR = Path(__file__).parents[1] / "data"
SEQUENCE_NAMES = ("TUD-Campus", "TUD-Stadtmitte")


class _IndependentCoverageFamily(mm.MetricFamily):
    """Track coverage with independent, non-bipartite matching semantics."""

    name = "independent_coverage"
    metric_names = ("independent_track_coverage", "mean_tracker_track_length")
    requirements = frozenset(("frame_iou", "trajectories"))
    display_names = {
        "independent_track_coverage": "IndTCOV",
        "mean_tracker_track_length": "MeanTL",
    }
    formatters = {"independent_track_coverage": "{:.1%}".format}

    def evaluate_sequence(self, sequence, intermediates):
        assert not sequence.ground_truth.frame_ids.flags.writeable
        assert not sequence.tracker.ids.flags.writeable
        assert "Confidence" in sequence.tracker.field_names
        assert not sequence.tracker.boxes.flags.writeable
        assert all(
            not frame.similarities.flags.writeable
            for frame in intermediates.frame_iou
        )
        lifespan = {}
        covered = {}
        for frame in intermediates.frame_iou:
            frame_coverage = np.any(frame.similarities >= 0.5, axis=1)
            for ground_truth_id, is_covered in zip(
                frame.ground_truth_ids,
                frame_coverage,
            ):
                lifespan[ground_truth_id] = lifespan.get(ground_truth_id, 0) + 1
                covered[ground_truth_id] = (
                    covered.get(ground_truth_id, 0) + int(is_covered)
                )
        coverage_sum = sum(
            covered.get(track_id, 0) / track_lifespan
            for track_id, track_lifespan in lifespan.items()
        )
        tracker_trajectories = intermediates.trajectories.tracker
        assert all(not trajectory.frame_ids.flags.writeable for trajectory in tracker_trajectories)
        return (
            coverage_sum,
            len(lifespan),
            len(sequence.tracker),
            len(tracker_trajectories),
        )

    def summarize(self, partial):
        coverage_sum, ground_truth_tracks, tracker_count, tracker_tracks = partial
        return {
            "independent_track_coverage": coverage_sum / max(1, ground_truth_tracks),
            "mean_tracker_track_length": tracker_count / max(1, tracker_tracks),
        }

    def combine(self, partials):
        return self.summarize(tuple(sum(values) for values in zip(*partials)))


class _TrackCoverageFamily(mm.MetricFamily):
    name = "track_coverage"
    metric_names = ("tcov",)
    requirements = frozenset(("clear_statistics",))
    display_names = {"tcov": "TCOV"}
    formatters = {"tcov": "{:.1%}".format}

    def evaluate_sequence(self, sequence, intermediates):
        del sequence
        coverage = intermediates.clear_statistics.track_coverage
        assert not coverage.flags.writeable
        return float(coverage.sum()), len(coverage)

    def summarize(self, partial):
        coverage_sum, track_count = partial
        return {"tcov": coverage_sum / max(1, track_count)}

    def combine(self, partials):
        return self.summarize(tuple(sum(values) for values in zip(*partials)))


class _ClearEventFamily(mm.MetricFamily):
    name = "clear_event_diagnostics"
    metric_names = (
        "event_switches",
        "event_transfers",
        "event_ascends",
        "event_migrates",
        "event_rows",
    )
    requirements = frozenset(("clear_events",))
    display_names = {
        "event_switches": "EventIDs",
        "event_transfers": "EventIDt",
        "event_ascends": "EventIDa",
        "event_migrates": "EventIDm",
        "event_rows": "Events",
    }
    formatters = {name: "{:d}".format for name in metric_names}

    def __init__(self, materialize_dataframe=False):
        self.materialize_dataframe = materialize_dataframe

    def evaluate_sequence(self, sequence, intermediates):
        events = intermediates.clear_events
        assert not events.frame_ids.flags.writeable
        assert not events.ground_truth_ids.flags.writeable
        if self.materialize_dataframe:
            assert list(events.df.columns) == ["Type", "OId", "HId", "D"]
            assert list(events.df.index.names) == ["FrameId", "Event"]
        return (
            int(np.count_nonzero(events.types == "SWITCH")),
            int(np.count_nonzero(events.types == "TRANSFER")),
            int(np.count_nonzero(events.types == "ASCEND")),
            int(np.count_nonzero(events.types == "MIGRATE")),
            len(events),
        )

    def summarize(self, partial):
        return dict(zip(self.metric_names, partial))

    def combine(self, partials):
        return self.summarize(tuple(sum(values) for values in zip(*partials)))


class _UndeclaredIntermediateFamily(mm.MetricFamily):
    name = "undeclared"
    metric_names = ("undeclared_value",)

    def evaluate_sequence(self, sequence, intermediates):
        return len(intermediates.frame_iou)

    def summarize(self, partial):
        return {"undeclared_value": partial}

    def combine(self, partials):
        return {"undeclared_value": sum(partials)}


class _InvalidRequirementFamily(_UndeclaredIntermediateFamily):
    name = "invalid_requirement"
    metric_names = ("invalid_requirement_value",)
    requirements = frozenset(("optical_flow",))


class _CollidingFamily(_UndeclaredIntermediateFamily):
    name = "collision"
    metric_names = ("mota",)


class _NonPicklableFamily(_IndependentCoverageFamily):
    name = "non_picklable"

    def __init__(self):
        self.callback = lambda value: value


def test_package_has_one_supported_metrics_entrypoint():
    assert mm.__all__ == ["evaluate_motchallenge", "MetricFamily"]
    assert {name for name in vars(mm) if not name.startswith("_")} == {
        "evaluate_motchallenge",
        "MetricFamily",
    }


def test_default_evaluation_does_not_import_pandas():
    script = """
import sys
import motmetrics as mm
assert 'pandas' not in sys.modules
summary = mm.evaluate_motchallenge({ground_truth!r}, {tracker!r})
str(summary)
assert summary['TUD-Campus', 'hota'] == summary['hota']['TUD-Campus']
assert summary['TUD-Campus']['hota'] == summary['TUD-Campus', 'hota']
assert 'pandas' not in sys.modules
""".format(
        ground_truth=str(DATA_DIR / "TUD-Campus" / "gt.txt"),
        tracker=str(DATA_DIR / "TUD-Campus" / "test.txt"),
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_compact_clear_events_do_not_import_pandas():
    script = """
import sys
import motmetrics as mm

class EventCount(mm.MetricFamily):
    name = 'event_count'
    metric_names = ('event_count',)
    requirements = frozenset(('clear_events',))

    def evaluate_sequence(self, sequence, intermediates):
        return len(intermediates.clear_events)

    def summarize(self, partial):
        return {{'event_count': partial}}

    def combine(self, partials):
        return {{'event_count': sum(partials)}}

assert 'pandas' not in sys.modules
summary = mm.evaluate_motchallenge(
    {ground_truth!r},
    {tracker!r},
    extra_metric_families=EventCount(),
)
assert summary['TUD-Campus', 'event_count'] > 0
str(summary)
assert 'pandas' not in sys.modules
""".format(
        ground_truth=str(DATA_DIR / "TUD-Campus" / "gt.txt"),
        tracker=str(DATA_DIR / "TUD-Campus" / "test.txt"),
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_default_path_does_not_construct_extension_views(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("default evaluation constructed extension state")

    monkeypatch.setattr(evaluation, "SequenceView", fail_if_called)
    monkeypatch.setattr(evaluation, "MetricContext", fail_if_called)
    monkeypatch.setattr(evaluation, "_ClearEventRecorder", fail_if_called)

    summary = mm.evaluate_motchallenge(
        DATA_DIR / "TUD-Campus" / "gt.txt",
        DATA_DIR / "TUD-Campus" / "test.txt",
    )
    assert summary["TUD-Campus", "hota"] == approx(0.3913974378451139)


def test_legacy_metric_modules_are_absent():
    removed_modules = (
        "motmetrics.__main__",
        "motmetrics.apps",
        "motmetrics.evaluation",
        "motmetrics.io",
        "motmetrics.lap",
        "motmetrics.metrics",
        "motmetrics.mot",
        "motmetrics.preprocess",
        "motmetrics.utils",
    )

    assert all(importlib.util.find_spec(module_name) is None for module_name in removed_modules)


def test_evaluate_motchallenge_files_returns_rich_summary():
    summary = mm.evaluate_motchallenge(
        DATA_DIR / "TUD-Campus" / "gt.txt",
        DATA_DIR / "TUD-Campus" / "test.txt",
    )

    assert isinstance(summary.df, pd.DataFrame)
    assert list(summary.df.index) == ["TUD-Campus"]
    assert "mota" in summary.df
    assert "hota" in summary.df
    assert summary.df.loc["TUD-Campus", "hota"] == approx(0.3913974378451139)
    assert summary.df.loc["TUD-Campus", "deta"] == approx(0.418047030142763)
    assert summary.df.loc["TUD-Campus", "assa"] == approx(0.36912068120832836)
    assert set(["detre", "detpr", "assre", "asspr", "loca", "owta"]).issubset(summary.df.columns)
    assert set(["moda", "smota", "mtr", "ptr", "mlr", "clr_f1", "fp_per_frame"]).issubset(summary.df.columns)
    assert str(summary) == summary.text
    assert "MOTA" in summary.text
    assert "HOTA" in summary.text


def test_motchallenge_preprocessing_suppresses_distractor_matches(tmp_path):
    ground_truth = tmp_path / "ground-truth.txt"
    tracker = tmp_path / "tracker.txt"
    ground_truth.write_text(
        "\n".join((
            "1,1,1,1,10,10,1,1,1",
            "1,2,21,1,10,10,0,8,1",
            "1,3,41,1,10,10,0,6,1",
        )),
        encoding="utf-8",
    )
    tracker.write_text(
        "\n".join((
            "1,10,1,1,10,10,1,1,1",
            "1,20,21,1,10,10,1,1,1",
            "1,30,41,1,10,10,1,1,1",
            "1,40,61,1,10,10,1,1,1",
        )),
        encoding="utf-8",
    )

    mot17 = mm.evaluate_motchallenge(ground_truth, tracker)
    mot20 = mm.evaluate_motchallenge(
        ground_truth,
        tracker,
        benchmark="MOT20",
    )
    without_preprocessing = mm.evaluate_motchallenge(
        ground_truth,
        tracker,
        benchmark="MOT15",
    )

    assert mot17[mot17.index[0], "num_unique_objects"] == 1
    assert mot17[mot17.index[0], "num_false_positives"] == 2
    assert mot20[mot20.index[0], "num_false_positives"] == 1
    assert without_preprocessing[
        without_preprocessing.index[0],
        "num_false_positives",
    ] == 3


def test_motchallenge_benchmark_defaults_are_loaded_from_yaml():
    mot15 = evaluation._load_benchmark_config("MOT15")
    mot17 = evaluation._load_benchmark_config("MOT17")
    mot20 = evaluation._load_benchmark_config("MOT20")
    sportsmot = evaluation._load_benchmark_config("SPORTSMOT")
    visdrone = evaluation._load_benchmark_config("VISDRONE")

    assert mot15["target_classes"] is None
    assert mot17["target_classes"] == (1,)
    assert mot17["distractor_classes"] == (2, 7, 8, 12)
    assert mot17["distractor_threshold"] == 0.5
    assert mot17["valid_classes"] == tuple(range(1, 14))
    assert mot17["tracker_max_class"] == 1
    assert mot20["distractor_classes"] == (2, 6, 7, 8, 12)
    assert sportsmot["target_classes"] == (1,)
    assert sportsmot["distractor_classes"] == ()
    assert visdrone["target_classes"] == (1, 4, 5, 6, 9)
    assert visdrone["distractor_classes"] == (0, 11)
    assert visdrone["class_aware"] is True
    assert visdrone["filter_tracker_classes"] is True
    assert visdrone["distractor_mode"] == "prediction_coverage"
    assert visdrone["suppress_target_ground_truth"] is True


def test_visdrone_profile_uses_class_aware_ignore_region_preprocessing(tmp_path):
    ground_truth = tmp_path / "visdrone-ground-truth.txt"
    tracker = tmp_path / "visdrone-tracker.txt"
    ground_truth.write_text(
        "\n".join((
            "1,1,1,1,10,10,1,1,0,0",
            "1,2,21,1,10,10,1,6,0,0",
            "1,90,21,1,10,10,0,0,0,0",
            "1,91,41,1,10,10,1,2,0,0",
            "2,1,1,1,10,10,1,4,0,0",
            "3,3,1,1,10,10,1,9,0,0",
            "4,4,1,1,10,10,1,5,0,0",
            "4,92,1,1,10,10,0,11,0,0",
        )),
        encoding="utf-8",
    )
    tracker.write_text(
        "\n".join((
            "1,10,1,1,10,10,1,1,-1,-1",
            "1,20,21,1,10,10,1,4,-1,-1",
            "1,30,41,1,10,10,1,2,-1,-1",
            "2,10,1,1,10,10,1,1,-1,-1",
            "3,30,1,1,10,10,1,9,-1,-1",
        )),
        encoding="utf-8",
    )

    summary = mm.evaluate_motchallenge(
        ground_truth,
        tracker,
        benchmark="VISDRONE",
        metrics=(
            "num_unique_objects",
            "num_detections",
            "num_misses",
            "num_false_positives",
        ),
    )
    row = summary.index[0]
    assert summary[row, "num_unique_objects"] == 3
    assert summary[row, "num_detections"] == 2
    assert summary[row, "num_misses"] == 1
    assert summary[row, "num_false_positives"] == 1


def test_sportsmot_profile_accepts_standard_single_class_data(tmp_path):
    ground_truth = tmp_path / "sportsmot-ground-truth.txt"
    tracker = tmp_path / "sportsmot-tracker.txt"
    ground_truth.write_text(
        "1,1,1,1,10,10,1,1,1\n",
        encoding="utf-8",
    )
    tracker.write_text(
        "1,10,1,1,10,10,1,-1,-1\n",
        encoding="utf-8",
    )

    summary = mm.evaluate_motchallenge(
        ground_truth,
        tracker,
        benchmark="SPORTSMOT",
        metrics=(
            "num_detections",
            "num_misses",
            "num_false_positives",
        ),
    )
    row = summary.index[0]
    assert summary[row, "num_detections"] == 1
    assert summary[row, "num_misses"] == 0
    assert summary[row, "num_false_positives"] == 0


def test_custom_class_preprocessing_supports_benchmark_specific_ids(tmp_path):
    ground_truth = tmp_path / "custom-ground-truth.txt"
    tracker = tmp_path / "custom-tracker.txt"
    ground_truth.write_text(
        "\n".join((
            "1,1,1,1,10,10,1,42,1",
            "1,2,21,1,10,10,1,43,1",
            "1,3,41,1,10,10,0,99,1",
            "1,4,61,1,10,10,0,100,1",
        )),
        encoding="utf-8",
    )
    tracker.write_text(
        "\n".join((
            "1,10,1,1,10,10,1,42,1",
            "1,20,21,1,10,10,1,43,1",
            "1,30,43,1,10,10,1,99,1",
            "1,40,61,1,10,10,1,100,1",
        )),
        encoding="utf-8",
    )

    custom = mm.evaluate_motchallenge(
        ground_truth,
        tracker,
        target_classes=(42, 43),
        distractor_classes=99,
    )
    no_suppression = mm.evaluate_motchallenge(
        ground_truth,
        tracker,
        target_classes=(42, 43),
        distractor_classes=(),
    )
    high_threshold = mm.evaluate_motchallenge(
        ground_truth,
        tracker,
        target_classes=(42, 43),
        distractor_classes=99,
        distractor_iou_threshold=0.75,
    )

    row = custom.index[0]
    assert custom[row, "num_unique_objects"] == 2
    assert custom[row, "num_false_positives"] == 1
    assert no_suppression[row, "num_false_positives"] == 2
    assert high_threshold[row, "num_false_positives"] == 2


def test_evaluate_motchallenge_rejects_invalid_benchmark():
    ground_truth = DATA_DIR / "TUD-Campus" / "gt.txt"
    tracker = DATA_DIR / "TUD-Campus" / "test.txt"

    with pytest.raises(ValueError, match="SPORTSMOT, or VISDRONE"):
        mm.evaluate_motchallenge(ground_truth, tracker, benchmark="MOT19")
    with pytest.raises(TypeError, match="benchmark must be"):
        mm.evaluate_motchallenge(ground_truth, tracker, benchmark=17)
    with pytest.raises(ValueError, match="must not overlap"):
        mm.evaluate_motchallenge(
            ground_truth,
            tracker,
            target_classes=(1, 2),
            distractor_classes=(2, 3),
        )
    with pytest.raises(TypeError, match="integer class IDs"):
        mm.evaluate_motchallenge(
            ground_truth,
            tracker,
            target_classes=(1, "person"),
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        mm.evaluate_motchallenge(
            ground_truth,
            tracker,
            distractor_iou_threshold=1.1,
        )


def test_summary_supports_native_metric_access():
    summary = mm.evaluate_motchallenge(DATA_DIR, DATA_DIR, progress=False)

    hota_by_sequence = summary["hota"]
    assert list(hota_by_sequence) == ["TUD-Campus", "TUD-Stadtmitte", "OVERALL"]
    assert hota_by_sequence["OVERALL"] == approx(0.3999570912884786)
    assert summary["OVERALL", "hota"] == approx(0.3999570912884786)
    assert summary["TUD-Campus", "hota"] == hota_by_sequence["TUD-Campus"]
    overall = summary["OVERALL"]
    assert list(overall) == summary.columns
    assert overall["hota"] == summary["OVERALL", "hota"]

    with pytest.raises(KeyError, match="Unknown summary row"):
        summary["missing", "hota"]
    with pytest.raises(KeyError, match="Unknown summary metric"):
        summary["OVERALL", "missing"]
    with pytest.raises(KeyError, match="Unknown summary row or metric"):
        summary["missing"]
    with pytest.raises(TypeError, match="row, metric name, or a \\(row, metric\\) pair"):
        summary[0]


def test_metrics_selects_core_and_hota_in_requested_order():
    summary = mm.evaluate_motchallenge(
        DATA_DIR / "TUD-Campus" / "gt.txt",
        DATA_DIR / "TUD-Campus" / "test.txt",
        metrics=["assa", "mota", "hota"],
    )

    assert list(summary.df.columns) == ["assa", "mota", "hota"]
    assert summary["TUD-Campus", "assa"] == approx(0.36912068120832836)
    assert summary["TUD-Campus", "mota"] == approx(0.5264623955431755)
    assert summary["TUD-Campus", "hota"] == approx(0.3913974378451139)


def test_metrics_without_hota_skips_hota_computation(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("HOTA was calculated despite not being selected")

    prepare_iou_sequence_data = evaluation._prepare_iou_sequence_data

    def assert_hota_is_not_prepared(*args, **kwargs):
        assert kwargs["compute_hota"] is False
        assert kwargs["retain_frame_iou"] is False
        return prepare_iou_sequence_data(*args, **kwargs)

    monkeypatch.setattr(
        evaluation,
        "_compute_prepared_hota_sequence_summary",
        fail_if_called,
    )
    monkeypatch.setattr(
        evaluation,
        "_prepare_iou_sequence_data",
        assert_hota_is_not_prepared,
    )
    summary = mm.evaluate_motchallenge(
        DATA_DIR / "TUD-Campus" / "gt.txt",
        DATA_DIR / "TUD-Campus" / "test.txt",
        metrics=["mota"],
    )

    assert summary.columns == ["mota"]
    assert summary["TUD-Campus", "mota"] == approx(0.5264623955431755)


def test_metrics_supports_hota_only_and_exclude_id():
    hota_only = mm.evaluate_motchallenge(
        DATA_DIR,
        DATA_DIR,
        metrics="hota",
        progress=False,
    )
    without_identity = mm.evaluate_motchallenge(
        DATA_DIR / "TUD-Campus" / "gt.txt",
        DATA_DIR / "TUD-Campus" / "test.txt",
        metrics=["idf1", "hota"],
        exclude_id=True,
    )

    assert hota_only.columns == ["hota"]
    assert hota_only["OVERALL", "hota"] == approx(0.3999570912884786)
    assert without_identity.columns == ["hota"]


def test_metrics_rejects_unknown_and_duplicate_names():
    ground_truth = DATA_DIR / "TUD-Campus" / "gt.txt"
    tracker = DATA_DIR / "TUD-Campus" / "test.txt"

    with pytest.raises(ValueError, match="Unknown metric: unknown"):
        mm.evaluate_motchallenge(ground_truth, tracker, metrics=["unknown"])
    with pytest.raises(ValueError, match="must not contain duplicate"):
        mm.evaluate_motchallenge(ground_truth, tracker, metrics=["hota", "hota"])


def test_evaluate_motchallenge_folders_returns_overall_summary(tmp_path):
    gt_root = tmp_path / "gt"
    test_root = tmp_path / "test"
    test_root.mkdir()

    for sequence_name in SEQUENCE_NAMES:
        source_dir = DATA_DIR / sequence_name
        sequence_gt_dir = gt_root / sequence_name / "gt"
        sequence_gt_dir.mkdir(parents=True)
        copyfile(source_dir / "gt.txt", sequence_gt_dir / "gt.txt")
        copyfile(source_dir / "test.txt", test_root / "{}.txt".format(sequence_name))
    (test_root / "person_summary.txt").write_text("not MOTChallenge data", encoding="utf-8")

    summary = mm.evaluate_motchallenge(gt_root, test_root)

    assert list(summary.df.index) == ["TUD-Campus", "TUD-Stadtmitte", "OVERALL"]
    assert set(evaluation.HOTA_SUMMARY_METRICS.values()).issubset(summary.df.columns)
    assert summary.df.loc["OVERALL", "hota"] == approx(0.3999570912884786)
    assert summary.df.loc["OVERALL", "deta"] == approx(0.3976832912424188)
    assert summary.df.loc["OVERALL", "assa"] == approx(0.4124495298453543)
    assert "IDF1" in summary.text
    assert "HOTA" in summary.text
    assert summary.df.to_csv().startswith(",idf1")


def test_evaluate_motchallenge_sequence_folders():
    summary = mm.evaluate_motchallenge(DATA_DIR / "TUD-Campus", DATA_DIR / "TUD-Campus")

    assert list(summary.df.index) == ["TUD-Campus", "OVERALL"]


def test_parallel_file_evaluation_matches_serial_results():
    serial = mm.evaluate_motchallenge(DATA_DIR, DATA_DIR, n_jobs=1).df
    parallel = mm.evaluate_motchallenge(DATA_DIR, DATA_DIR, n_jobs=2).df

    pd.testing.assert_frame_equal(parallel, serial)


def test_custom_metric_family_owns_matching_and_overall_aggregation():
    family = _IndependentCoverageFamily()
    default = mm.evaluate_motchallenge(DATA_DIR, DATA_DIR, progress=False)
    summary = mm.evaluate_motchallenge(
        DATA_DIR,
        DATA_DIR,
        extra_metric_families=family,
        progress=False,
    )

    assert summary.columns[-2:] == list(family.metric_names)
    assert "IndTCOV" in summary.text
    for row_name in summary.index:
        for metric_name in default.columns:
            assert summary[row_name, metric_name] == default[row_name, metric_name]

    sequence_partials = []
    for sequence_name in SEQUENCE_NAMES:
        ground_truth = evaluation.io.loadtxt(
            DATA_DIR / sequence_name / "gt.txt",
            min_confidence=1,
        )
        tracker = evaluation.io.loadtxt(DATA_DIR / sequence_name / "test.txt")
        tracker_track_count = len(np.unique(tracker.ids))
        ground_truth_track_count = len(np.unique(ground_truth.ids))
        track_coverage = summary[sequence_name, "independent_track_coverage"]
        sequence_partials.append((
            track_coverage * ground_truth_track_count,
            ground_truth_track_count,
            len(tracker),
            tracker_track_count,
        ))
    expected_overall = family.combine(sequence_partials)
    assert summary["OVERALL", "independent_track_coverage"] == approx(
        expected_overall["independent_track_coverage"]
    )
    assert summary["OVERALL", "mean_tracker_track_length"] == approx(
        expected_overall["mean_tracker_track_length"]
    )


def test_custom_metric_family_matches_between_serial_and_parallel():
    families = (
        _IndependentCoverageFamily(),
        _TrackCoverageFamily(),
        _ClearEventFamily(),
    )
    serial = mm.evaluate_motchallenge(
        DATA_DIR,
        DATA_DIR,
        n_jobs=1,
        progress=False,
        extra_metric_families=families,
    ).df
    parallel = mm.evaluate_motchallenge(
        DATA_DIR,
        DATA_DIR,
        n_jobs=2,
        progress=False,
        extra_metric_families=families,
    ).df

    pd.testing.assert_frame_equal(parallel, serial)


def test_clear_statistics_reuses_compact_match_counts(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("clear statistics constructed event history")

    monkeypatch.setattr(evaluation, "_ClearEventRecorder", fail_if_called)
    summary = mm.evaluate_motchallenge(
        DATA_DIR / "TUD-Campus" / "gt.txt",
        DATA_DIR / "TUD-Campus" / "test.txt",
        extra_metric_families=_TrackCoverageFamily(),
    )

    assert summary["TUD-Campus", "tcov"] == approx(
        0.5794012687234518
    )


def test_clear_events_are_opt_in_and_match_fast_clear_counts():
    summary = mm.evaluate_motchallenge(
        DATA_DIR / "TUD-Campus" / "gt.txt",
        DATA_DIR / "TUD-Campus" / "test.txt",
        extra_metric_families=_ClearEventFamily(materialize_dataframe=True),
    )

    assert summary["TUD-Campus", "event_switches"] == summary[
        "TUD-Campus",
        "num_switches",
    ]
    assert summary["TUD-Campus", "event_transfers"] == summary[
        "TUD-Campus",
        "num_transfer",
    ]
    assert summary["TUD-Campus", "event_ascends"] == summary[
        "TUD-Campus",
        "num_ascend",
    ]
    assert summary["TUD-Campus", "event_migrates"] == summary[
        "TUD-Campus",
        "num_migrate",
    ]
    assert summary["TUD-Campus", "event_rows"] > len(
        evaluation.io.loadtxt(DATA_DIR / "TUD-Campus" / "gt.txt")
    )


def test_metric_family_cannot_access_undeclared_intermediate():
    with pytest.raises(RuntimeError, match="must declare the 'frame_iou' requirement"):
        mm.evaluate_motchallenge(
            DATA_DIR / "TUD-Campus" / "gt.txt",
            DATA_DIR / "TUD-Campus" / "test.txt",
            extra_metric_families=_UndeclaredIntermediateFamily(),
        )


def test_metric_family_validation_fails_before_evaluation():
    with pytest.raises(ValueError, match="unsupported intermediates: optical_flow"):
        mm.evaluate_motchallenge(
            DATA_DIR,
            DATA_DIR,
            extra_metric_families=_InvalidRequirementFamily(),
        )
    with pytest.raises(ValueError, match="reuses existing metric names: mota"):
        mm.evaluate_motchallenge(
            DATA_DIR,
            DATA_DIR,
            extra_metric_families=_CollidingFamily(),
        )
    with pytest.raises(TypeError, match="must be picklable"):
        mm.evaluate_motchallenge(
            DATA_DIR,
            DATA_DIR,
            n_jobs=2,
            extra_metric_families=_NonPicklableFamily(),
        )


def test_parallel_progress_renders_one_terminal_row_per_sequence(capsys):
    mm.evaluate_motchallenge(DATA_DIR, DATA_DIR, n_jobs=2, progress=True)

    terminal_output = capsys.readouterr().err
    final_render = terminal_output.rsplit("\x1b[2A", 1)[1]
    assert final_render.count("\x1b[2K") == len(SEQUENCE_NAMES)
    assert final_render.count("done") == len(SEQUENCE_NAMES)
    assert all(sequence_name in final_render for sequence_name in SEQUENCE_NAMES)
    assert terminal_output.endswith("\x1b[?25h")


def test_evaluate_motchallenge_rejects_mixed_file_and_folder_inputs():
    gt_file = DATA_DIR / "TUD-Campus" / "gt.txt"

    try:
        mm.evaluate_motchallenge(gt_file, DATA_DIR / "TUD-Campus")
    except ValueError as exc:
        assert "both be files or both be folders" in str(exc)
    else:
        raise AssertionError("Expected mixed file/folder inputs to fail.")


def test_validate_n_jobs_defaults_and_flag_overrides(monkeypatch):
    monkeypatch.setattr(evaluation.os, "cpu_count", lambda: 8)
    # Default for multi-sequence evaluation when n_jobs is None -> min(num_tasks, max(1, cpu_count - 2))
    assert evaluation._validate_n_jobs(None, num_tasks=10) == 6
    assert evaluation._validate_n_jobs(None, num_tasks=4) == 4
    assert evaluation._validate_n_jobs(None, num_tasks=2) == 2
    # Default for single-sequence evaluation when n_jobs is None -> 1
    assert evaluation._validate_n_jobs(None, num_tasks=1) == 1

    # Explicit flag overrides default behavior
    assert evaluation._validate_n_jobs(12, num_tasks=4) == 12
    assert evaluation._validate_n_jobs(8, num_tasks=4) == 8
    assert evaluation._validate_n_jobs(4, num_tasks=4) == 4
    assert evaluation._validate_n_jobs(1, num_tasks=4) == 1

    monkeypatch.setattr(evaluation.os, "cpu_count", lambda: 2)
    assert evaluation._validate_n_jobs(None, num_tasks=10) == 1
    assert evaluation._validate_n_jobs(4, num_tasks=10) == 4


def test_hota_is_invariant_to_zero_tracker_id():
    hota_alphas = np.array([0.25, 0.5, 0.75])
    gt = _mot_dataframe([
        [1, 1, 24, 36, 10, 10],
        [1, 2, 36, 0, 10, 10],
        [2, 1, 12, 24, 10, 10],
        [2, 2, 24, 36, 10, 10],
        [3, 1, 12, 12, 10, 10],
        [3, 2, 36, 0, 10, 10],
        [4, 1, 36, 36, 10, 10],
        [4, 2, 0, 24, 10, 10],
        [5, 2, 12, 24, 10, 10],
        [6, 1, 24, 0, 10, 10],
        [7, 1, 24, 24, 10, 10],
        [7, 2, 36, 12, 10, 10],
    ])
    test = _mot_dataframe([
        [1, 0, 12.090215403896158, 38.0103228683318, 10, 10],
        [1, 2, 36.535330512240094, 12.158121373496847, 10, 10],
        [2, 0, 22.065693390322537, 21.237397443312403, 10, 10],
        [2, 1, 36.40689653823255, 24.23512662993634, 10, 10],
        [2, 2, 35.191960656230044, 11.927248581898391, 10, 10],
        [3, 0, 37.59134793507912, 22.788697987002156, 10, 10],
        [3, 1, 23.124599350885045, 35.83244707562376, 10, 10],
        [4, 0, 0.17903103854487185, 23.03779440883917, 10, 10],
        [4, 1, 22.201066646842165, 24.111774343157194, 10, 10],
        [4, 2, 1.024365400792809, -0.09977598022412335, 10, 10],
        [5, 0, 24.190902616838745, 22.21920820822479, 10, 10],
        [5, 1, 1.3481458081506117, 13.717833011181199, 10, 10],
        [5, 2, 0.9703551338601327, 21.01137032373826, 10, 10],
        [6, 0, 23.44713615885006, 35.62470689922311, 10, 10],
        [6, 1, 35.544479417452905, 36.5288836009279, 10, 10],
        [7, 0, 23.9637845804851, 1.0025715349010158, 10, 10],
        [7, 1, 11.99190065899256, 12.87507353127062, 10, 10],
        [7, 2, 33.467693823950185, -3.0529934174098985, 10, 10],
    ])

    shifted_test = evaluation.io._SequenceData(
        test.frame_ids,
        test.ids + 100,
        test._fields,
    )

    original = evaluation._compute_prepared_hota_sequence_summary(
        evaluation._prepare_iou_sequence_data(gt, test, 0.5),
        hota_alphas,
    )
    shifted = evaluation._compute_prepared_hota_sequence_summary(
        evaluation._prepare_iou_sequence_data(gt, shifted_test, 0.5),
        hota_alphas,
    )
    for metric in evaluation.HOTA_SUMMARY_METRICS:
        np.testing.assert_allclose(original[metric], shifted[metric], rtol=0, atol=0)


def test_frame_array_grouping_handles_duplicate_id_counts():
    tracker = _mot_dataframe([
        [1, -1, 0, 0, 10, 10],
        [1, -1, 20, 20, 10, 10],
        [2, -1, 0, 0, 10, 10],
        [2, -1, 20, 20, 10, 10],
    ])

    groups, counts = evaluation._group_frame_arrays(
        tracker,
        np.asarray([-1]),
        ["X", "Y", "Width", "Height"],
    )

    assert len(groups) == 2
    np.testing.assert_array_equal(counts, [4])


def _mot_dataframe(rows):
    values = np.asarray(rows, dtype=float)
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    return evaluation.io._SequenceData(
        values[:, 0],
        values[:, 1],
        {
            "X": values[:, 2],
            "Y": values[:, 3],
            "Width": values[:, 4],
            "Height": values[:, 5],
        },
    )
