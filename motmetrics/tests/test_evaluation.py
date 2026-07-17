from pathlib import Path
from shutil import copyfile

import pandas as pd
from pytest import approx

import motmetrics as mm

DATA_DIR = Path(__file__).parents[1] / "data"
SEQUENCE_NAMES = ("TUD-Campus", "TUD-Stadtmitte")


def test_evaluate_motchallenge_files_returns_rich_summary():
    summary = mm.evaluate_motchallenge(
        DATA_DIR / "TUD-Campus" / "gt.txt",
        DATA_DIR / "TUD-Campus" / "test.txt",
    )

    assert isinstance(summary.df, pd.DataFrame)
    assert summary.dataframe is summary.df
    assert list(summary.df.index) == ["TUD-Campus"]
    assert "mota" in summary
    assert "hota" in summary
    assert summary["mota"].equals(summary.df["mota"])
    assert summary.mota.equals(summary.df["mota"])
    assert summary.hota.equals(summary.df["hota"])
    assert summary.df.loc["TUD-Campus", "hota"] == approx(0.3913974378451139)
    assert summary.df.loc["TUD-Campus", "deta"] == approx(0.418047030142763)
    assert summary.df.loc["TUD-Campus", "assa"] == approx(0.36912068120832836)
    assert summary.loc["TUD-Campus", "mota"] == summary.df.loc["TUD-Campus", "mota"]
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

    summary = mm.evaluate_motchallenge(gt_root, test_root)

    assert list(summary.df.index) == ["TUD-Campus", "TUD-Stadtmitte", "OVERALL"]
    assert set(["hota", "deta", "assa"]).issubset(summary.df.columns)
    assert summary.df.loc["OVERALL", "hota"] == approx(0.3999570912884786)
    assert summary.df.loc["OVERALL", "deta"] == approx(0.3976832912424188)
    assert summary.df.loc["OVERALL", "assa"] == approx(0.4124495298453543)
    assert "IDF1" in summary.text
    assert "HOTA" in summary.text
    assert summary.to_csv().startswith(",idf1")


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


def test_evaluate_motchallenge_rejects_mixed_file_and_folder_inputs():
    gt_file = DATA_DIR / "TUD-Campus" / "gt.txt"

    try:
        mm.evaluate_motchallenge(gt_file, DATA_DIR / "TUD-Campus")
    except ValueError as exc:
        assert "both be files or both be folders" in str(exc)
    else:
        raise AssertionError("Expected mixed file/folder inputs to fail.")
