# Release procedure

Releases are built on demand from the current `develop` branch by GitHub
Actions. A maintainer chooses the version bump. Building, publishing, committing
the production version, tagging, and creating the GitHub release are automated.

The current version is stored only in `[project].version` in `pyproject.toml`.
The workflow reads that value and calculates the selected patch, minor, or major
bump.

PyPI uses Trusted Publishing, so no package index token is stored in GitHub.

## One-time repository setup

1. Create a GitHub environment named `pypi`.
2. In the PyPI settings for `motmetrics`, add a GitHub Trusted Publisher with:

   - owner: `cheind`
   - repository: `py-motmetrics`
   - workflow: `publish-to-pypi.yml`
   - environment: `pypi`

3. Restrict the environment to deployments from `develop` and require
   maintainer approval.
4. Add `RELEASE_PAT` as a secret on the protected `pypi` environment. Use a
   fine-grained token limited to this repository with Contents read/write
   access. A repository secret also works, but has broader workflow scope.
5. Ensure the token owner can push the version commit directly to `develop`
   and create `v*` tags. If branch or tag rulesets are enabled, add that identity
   to their bypass lists. The workflow falls back to `GITHUB_TOKEN` when
   repository rules permit that token to push.

## Normal release

1. Open **Actions > Publish to PyPI > Run workflow** and select `develop` in
   the branch selector.
2. Select `patch`, `minor`, or `major`.
3. The workflow checks out `develop`, builds and checks the wheel and source
   distribution, and smoke-tests both artifacts. The regular Python package CI
   workflow remains responsible for pytest coverage.
4. Approve the protected `pypi` environment deployment. The workflow
   commits the selected version to `develop`, publishes the tested artifacts,
   creates the `vX.Y.Z` tag and GitHub release, and attaches both distributions.

If `develop` changes while the release is being tested, production publishing
stops before committing or uploading anything. Start a new workflow run from
the updated branch.

If the version commit succeeds but publishing or release creation later fails,
rerun the failed job. The workflow recognizes its existing version-only commit,
skips package files already accepted by the index, and resumes tag and release
creation.

## Verify a release

Install the new version without using a local package cache:

    python -m pip install --no-cache-dir motmetrics==X.Y.Z
    python -c "import motmetrics; print(motmetrics.__version__)"

PyPI files are immutable. If publishing fails after an artifact was accepted,
increment the version and publish a new release; never reuse or move a release
tag.
