from importlib.metadata import distribution, version

import motmetrics as mm


def test_package_version_matches_distribution_metadata():
    assert mm.__version__ == version("motmetrics")


def test_distribution_has_no_command_entrypoints():
    console_scripts = [
        entry_point
        for entry_point in distribution("motmetrics").entry_points
        if entry_point.group == "console_scripts"
    ]

    assert console_scripts == []
