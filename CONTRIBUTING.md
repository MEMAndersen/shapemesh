# Contributing

## Releasing

This checklist is for maintainers preparing and publishing a release. Start from
a clean `main` and do not push release commits or tags until they have been
reviewed.

### 1. Prepare

- [ ] Confirm `main` is up to date and the working tree is clean.
- [ ] Bump `__version__` in `src/shapemesh/__init__.py`. This is the only version
      literal; `pyproject.toml` reads it dynamically. `uv version` cannot get or
      set dynamic versions, so edit the file directly.
- [ ] Update `CHANGELOG.md` following Common Changelog and `AGENTS.md`: use the
      release date, merge related changes, link the relevant PRs or commits, and
      credit the author. Do not add an `Unreleased` section.
- [ ] Run the local gates: `uv lock --check`, Ruff lint and format checks,
      Pyright, and the full test suite.
- [ ] Push the preparation commits and wait for all CI jobs to pass.

### 2. Build and verify

- [ ] Build from an empty `dist/` on the final release commit:
      `uv build`
- [ ] Check the archives with Twine:
      `uvx twine check --strict dist/*.whl dist/*.tar.gz`
- [ ] Inspect both archives. Confirm the version, `py.typed`, `COPYING`,
      `COPYING.LESSER`, `README.md`, `CHANGELOG.md`, examples, and project URLs.
      Confirm that no stale API, virtual environment, cache, or build directory
      was packaged. Check the sdist file count and size are reasonable.
- [ ] Record the SHA-256 hashes of the wheel and sdist:

```powershell
Get-ChildItem dist -File |
      Where-Object { $_.Name -match '\.(whl|tar\.gz)$' } |
      Get-FileHash -Algorithm SHA256 |
      Format-List Path, Hash
```

- [ ] Install the wheel in a clean environment outside the repository and run
      the release smoke test:

```powershell
$repo = (Get-Location).Path
$check = Join-Path $env:TEMP "shapemesh-release-check"
$python = Join-Path $check ".venv\Scripts\python.exe"
$wheel = (Get-ChildItem "$repo\dist\*.whl").FullName

Remove-Item $check -Recurse -Force -ErrorAction SilentlyContinue
uv venv "$check\.venv" --python 3.12
uv pip install --python $python $wheel
& $python "$repo\scripts\release_smoke.py" --expect-version X.Y.Z
```

### 3. Tag

- [ ] Commit any final changes and confirm CI is green on the exact commit that
      produced the artifacts.
- [ ] Create an annotated tag on that commit:

```text
git tag -a vX.Y.Z -m "shapemesh X.Y.Z"
git push origin vX.Y.Z
```

- [ ] Do not add commits or rebuild after tagging without repeating the build
      and verification step.

### 4. Publish

- [ ] Optional but recommended for a first release: publish to TestPyPI and
      install from there in a clean environment. When installing from TestPyPI, add PyPI as an extra index so dependencies
      resolve normally.

```text
uv publish --publish-url https://test.pypi.org/legacy/ dist/*.whl dist/*.tar.gz
```

      

- [ ] Set `UV_PUBLISH_TOKEN` in the environment, then publish to PyPI:

```text
uv publish dist/*.whl dist/*.tar.gz
```

- [ ] Verify the version, files, metadata, and hashes on PyPI.
- [ ] Create a GitHub release from the pushed tag and use the `CHANGELOG.md`
      entry as the release notes.

### 5. Finish

- [ ] Install `shapemesh==X.Y.Z` from PyPI in a clean environment outside the
      repository and run a smoke test.
- [ ] Confirm the tag, GitHub release, and PyPI files all refer to the same
      release commit and version.
- [ ] Remove local scratch environments and build output.
