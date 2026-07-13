from importlib.metadata import version

import motmetrics as mm


def test_package_version_matches_distribution_metadata():
    assert mm.__version__ == version("motmetrics")
