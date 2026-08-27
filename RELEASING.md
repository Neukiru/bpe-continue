# Releasing

Releases are built and published by GitHub Actions. Nothing is uploaded from a laptop, and
there is no PyPI API token anywhere: authentication uses [Trusted
Publishing](https://docs.pypi.org/trusted-publishers/), where PyPI verifies that the upload
came from this repository's workflow.

## One-time setup

### 1. Push the repository

```sh
git remote add origin git@github.com:<you>/bpe-continue.git
git push -u origin main
```

### 2. Register the trusted publisher

Do this on **both** indexes. Neither project exists yet, so use the "pending publisher" form,
which claims the name on first upload:

- TestPyPI — <https://test.pypi.org/manage/account/publishing/>
- PyPI — <https://pypi.org/manage/account/publishing/>

| Field | Value |
| --- | --- |
| PyPI project name | `bpe-continue` |
| Owner | your GitHub account or org |
| Repository name | `bpe-continue` |
| Workflow name | `ci.yml` |
| Environment name | `testpypi` on TestPyPI, `pypi` on PyPI |

### 3. Create the GitHub environments

In **Settings → Environments**, add `testpypi` and `pypi`. They can be empty — they exist so
the trusted-publisher claim is scoped to them. Adding a required reviewer to `pypi` gives you a
manual approval step before anything reaches production.

## Cutting a release

### 1. Rehearse on TestPyPI

Run the **CI** workflow manually (Actions → CI → Run workflow) with **Publish to TestPyPI**
checked. Then install what it produced into a clean environment and confirm it works:

```sh
python -m venv /tmp/check && /tmp/check/bin/pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    bpe-continue
```

The extra index is needed because `tokenizers` is not on TestPyPI.

### 2. Publish

```sh
git tag v0.1.0
git push origin v0.1.0
```

The tag triggers the full matrix — tests on three platforms, wheels for Linux
x86_64/aarch64, macOS x86_64/arm64 and Windows x64, plus an sdist — and publishes to PyPI only
if every job passes.

## Wheels

The extension is built against Python's stable ABI (`abi3-py39` in `Cargo.toml`), so each
platform needs one wheel rather than one per Python version, and wheels keep working on Python
releases that postdate them. Raising the floor means changing `abi3-pyXY` and `requires-python`
together.

To build locally instead of in CI — Linux wheels need Docker:

```sh
./scripts/build-wheels.sh          # this machine + linux x86_64
./scripts/build-wheels.sh linux    # linux x86_64 only
```

## Version numbers

The version lives in two places, which must agree:

- `pyproject.toml` → `project.version`
- `Cargo.toml` → `package.version`

A version can never be reused on PyPI, even after a yank, so bump before re-releasing.

## If something goes wrong

- **Wrong metadata in a published release** — bump the patch version and release again. Yank
  the bad one from the PyPI project page; yanking hides it from new installs while leaving
  existing pinned installs working.
- **A platform's wheel fails to build** — the publish job depends on every wheel job, so
  nothing is uploaded. Fix and re-tag with a new version.
- **No wheel for a user's platform** — they fall back to the sdist, which compiles the vendored
  Rust core and needs a Rust toolchain. Adding a target to the `wheels` matrix avoids that.
