# Release procedure

Releases are built on demand from the current `develop` branch by GitHub
Actions. A maintainer chooses the version bump and package repository. Testing,
building, publishing, committing the production version, tagging, and creating
the GitHub release are automated.

The current version is stored only in `[project].version` in `pyproject.toml`.
The workflow reads that value and calculates the selected patch, minor, or major
bump.

PyPI and TestPyPI use Trusted Publishing, so no package index tokens are stored
in GitHub.

## One-time repository setup

1. Create GitHub environments named `pypi` and `testpypi`.
2. In the PyPI settings for `motmetrics`, add a GitHub Trusted Publisher with:

   - owner: `cheind`
   - repository: `py-motmetrics`
   - workflow: `publish-to-pypi.yml`
   - environment: `pypi`

3. In the TestPyPI settings for `motmetrics`, add the same publisher using the
   `testpypi` environment.
4. Require maintainer approval on the `pypi` environment. The `testpypi`
   environment can remain unprotected for convenient release-candidate tests.
5. Add a `RELEASE_PAT` repository secret with Contents read/write access and
   permission to push the version commit to `develop`. The workflow falls back
   to `GITHUB_TOKEN` when repository rules permit that token to push.

## Normal release

1. Open **Actions > Publish to PyPI > Run workflow**.
2. Select `patch`, `minor`, or `major` and choose `testpypi` or `pypi`.
3. The workflow checks out `develop`, runs the complete Python matrix, builds
   and checks the wheel and source distribution, and smoke-tests both artifacts.
4. For TestPyPI, the candidate is published without changing `develop`.
5. For PyPI, approve the protected `pypi` environment deployment. The workflow
   commits the selected version to `develop`, publishes the tested artifacts,
   creates the `vX.Y.Z` tag and GitHub release, and attaches both distributions.

If `develop` changes while the release is being tested, production publishing
stops before committing or uploading anything. Start a new workflow run from
the updated branch.

## Verify a release

Install the new version without using a local package cache:

    python -m pip install --no-cache-dir motmetrics==X.Y.Z
    python -c "import motmetrics; print(motmetrics.__version__)"

PyPI files are immutable. If publishing fails after an artifact was accepted,
increment the version and publish a new release; never reuse or move a release
tag.
