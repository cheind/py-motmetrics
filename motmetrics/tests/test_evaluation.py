from pathlib import Path
from shutil import copyfile

import pandas as pd

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


def test_evaluate_motchallenge_treats_negative_one_gt_confidence_as_active(tmp_path):
    gt_file = tmp_path / "gt.txt"
    test_file = tmp_path / "test.txt"
    gt_file.write_text(
        "1,1,100,100,50,100,-1,-1,-1,-1\n"
        "1,2,300,300,50,100,0,-1,-1,-1\n",
        encoding="utf-8",
    )
    test_file.write_text(
        "1,1,100,100,50,100,-1,-1,-1,-1\n",
        encoding="utf-8",
    )

    summary = mm.evaluate_motchallenge(gt_file, test_file)

    assert summary.df.iloc[0]["num_unique_objects"] == 1
    assert summary.df.iloc[0]["num_false_positives"] == 0
    assert summary.df.iloc[0]["num_misses"] == 0
    assert summary.df.iloc[0]["mota"] == 1.0
    assert summary.df.iloc[0]["hota"] == 1.0


def test_evaluate_motchallenge_explicit_gt_min_confidence_still_filters(tmp_path):
    gt_file = tmp_path / "gt.txt"
    test_file = tmp_path / "test.txt"
    gt_file.write_text(
        "1,1,100,100,50,100,-1,-1,-1,-1\n"
        "1,2,300,300,50,100,1,-1,-1,-1\n",
        encoding="utf-8",
    )
    test_file.write_text(
        "1,2,300,300,50,100,1,-1,-1,-1\n",
        encoding="utf-8",
    )

    summary = mm.evaluate_motchallenge(gt_file, test_file, gt_min_confidence=1)

    assert summary.df.iloc[0]["num_unique_objects"] == 1
    assert summary.df.iloc[0]["mota"] == 1.0
