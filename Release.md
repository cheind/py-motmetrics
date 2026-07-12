# Release procedure

Releases are built from Git tags and published by GitHub Actions using PyPI
Trusted Publishing. Maintainers do not need to store a long-lived PyPI token in
GitHub.

## One-time repository setup

1. In the PyPI settings for `motmetrics`, add a GitHub Trusted Publisher with:
   - owner: `cheind`
   - repository: `py-motmetrics`
   - workflow: `publish-to-pypi.yml`
   - environment: `pypi`
2. Create a protected GitHub environment named `pypi` and require maintainer
   approval before deployment.
3. Protect `master` and require the Python package workflow to pass before a
   release pull request can merge.

## Prepare a release

1. Create a short `release/<version>` branch from an up-to-date `master`.
2. Update `motmetrics.__version__` in `motmetrics/__init__.py` using a valid PEP
   440 version without a leading `v`.
3. Update the changelog or release notes with user-visible changes.
4. Run the supported-version CI matrix and local checks:

       python -m pip install --upgrade build twine
       python -m build
       python -m twine check dist/*
       uv venv
       uv pip install --group dev
       uv run --no-project --no-sync pytest

5. Install the generated wheel in a clean environment and smoke-test the
   import and version.
6. Open a pull request to `master`, obtain review, and merge only after all
   required checks pass.

## Publish

1. From the merged `master` commit, create and push a signed tag whose name is
   the package version prefixed with `v`:

       git tag -s v1.5.0 -m "Release 1.5.0"
       git push origin v1.5.0

2. The `Publish to PyPI` workflow builds one source distribution and one
   universal wheel with a current Python and setuptools 77 or newer, checks
   their metadata, verifies the artifact version
   against the tag, smoke-tests the wheel, and waits for approval on the `pypi`
   environment before publishing.
3. After the workflow succeeds, create GitHub release notes for the same tag
   and verify the published package:

       python -m pip install --no-cache-dir motmetrics==1.5.0
       python -c "import motmetrics; print(motmetrics.__version__)"

PyPI files are immutable. If publishing fails after any artifact was accepted,
increment the version and publish a new release; never reuse or move a release
tag.
