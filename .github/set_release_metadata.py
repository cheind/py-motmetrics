"""Synchronize static package and citation metadata for a release."""

import os
import re
from pathlib import Path


def _replace_one(path, pattern, replacement, label):
    content = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        pattern,
        replacement,
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit("Could not update {}".format(label))
    path.write_text(updated, encoding="utf-8")


release_version = os.environ["RELEASE_VERSION"]
release_date = os.environ["RELEASE_DATE"]
if re.fullmatch(r"\d+\.\d+\.\d+", release_version) is None:
    raise SystemExit("RELEASE_VERSION must use X.Y.Z format")
if re.fullmatch(r"\d{4}-\d{2}-\d{2}", release_date) is None:
    raise SystemExit("RELEASE_DATE must use YYYY-MM-DD format")

_replace_one(
    Path("pyproject.toml"),
    r'^version = "\d+\.\d+\.\d+"$',
    'version = "{}"'.format(release_version),
    "project.version",
)
_replace_one(
    Path("CITATION.cff"),
    r'^version: "\d+\.\d+\.\d+"$',
    'version: "{}"'.format(release_version),
    "citation version",
)
_replace_one(
    Path("CITATION.cff"),
    r'^date-released: "\d{4}-\d{2}-\d{2}"$',
    'date-released: "{}"'.format(release_date),
    "citation release date",
)
