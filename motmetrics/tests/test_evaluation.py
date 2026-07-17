import importlib.util
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
from pytest import approx

import motmetrics as mm
import motmetrics._evaluation as evaluation

DATA_DIR = Path(__file__).parents[1] / "data"
SEQUENCE_NAMES = ("TUD-Campus", "TUD-Stadtmitte")


def test_package_has_one_supported_metrics_entrypoint():
    assert mm.__all__ == ["evaluate_motchallenge"]
    assert {name for name in vars(mm) if not name.startswith("_")} == {
        "evaluate_motchallenge"
    }


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
    assert str(summary) == summary.text
    assert "MOTA" in summary.text
    assert "HOTA" in summary.text


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
    assert set(["hota", "deta", "assa"]).issubset(summary.df.columns)
    assert summary.df.loc["OVERALL", "hota"] == approx(0.3999570912884786)
    assert summary.df.loc["OVERALL", "deta"] == approx(0.3976832912424188)
    assert summary.df.loc["OVERALL", "assa"] == approx(0.4124495298453543)
    assert "IDF1" in summary.text
    assert "HOTA" in summary.text
    assert summary.df.to_csv().startswith(",idf1")


def test_evaluate_motchallenge_can_skip_hota():
    summary = mm.evaluate_motchallenge(
        DATA_DIR / "TUD-Campus" / "gt.txt",
        DATA_DIR / "TUD-Campus" / "test.txt",
        include_hota=False,
    )

    assert "hota" not in summary.df.columns
    assert "HOTA" not in summary.text


def test_evaluate_motchallenge_sequence_folders():
    summary = mm.evaluate_motchallenge(DATA_DIR / "TUD-Campus", DATA_DIR / "TUD-Campus")

    assert list(summary.df.index) == ["TUD-Campus", "OVERALL"]


def test_parallel_file_evaluation_matches_serial_results():
    serial = mm.evaluate_motchallenge(DATA_DIR, DATA_DIR, n_jobs=1).df
    parallel = mm.evaluate_motchallenge(DATA_DIR, DATA_DIR, n_jobs=2).df

    pd.testing.assert_frame_equal(parallel, serial)


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

    shifted_test = test.reset_index()
    shifted_test["Id"] += 100
    shifted_test = shifted_test.set_index(["FrameId", "Id"]).sort_index()

    original = evaluation._compute_prepared_hota_sequence_summary(
        evaluation._prepare_iou_sequence_data(gt, test, 0.5),
        hota_alphas,
    )
    shifted = evaluation._compute_prepared_hota_sequence_summary(
        evaluation._prepare_iou_sequence_data(gt, shifted_test, 0.5),
        hota_alphas,
    )
    for metric in ("hota_alpha", "deta_alpha", "assa_alpha"):
        np.testing.assert_allclose(original[metric], shifted[metric], rtol=0, atol=0)


def test_frame_array_grouping_handles_duplicate_id_counts():
    tracker = _mot_dataframe([
        [1, -1, 0, 0, 10, 10],
        [1, -1, 20, 20, 10, 10],
        [2, -1, 0, 0, 10, 10],
        [2, -1, 20, 20, 10, 10],
    ])

    groups, counts = evaluation._group_frame_arrays(tracker, pd.Index([-1]))

    assert len(groups) == 2
    np.testing.assert_array_equal(counts, [4])


def _mot_dataframe(rows):
    df = pd.DataFrame(rows, columns=["FrameId", "Id", "X", "Y", "Width", "Height"])
    return df.set_index(["FrameId", "Id"]).sort_index()
