# QGYBJ.jl

[![CI](https://github.com/subhk/QGYBJ.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/subhk/QGYBJ.jl/actions/workflows/ci.yml)
[![Docs: Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/QGYBJ.jl/stable)
[![Docs: Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/QGYBJ.jl/dev)

Minimal Julia package skeleton.

## Usage

- Develop locally: in Julia REPL, run `] activate .` then `] test` to run tests.
- Use in another project: `] dev path/to/QGYBJ.jl` or add once published.

## Structure

- `Project.toml`: package metadata.
- `src/QGYBJ.jl`: module entry point.
- `test/runtests.jl`: basic test scaffold.

## CI

- GitHub Actions runs tests on Linux, macOS, and Windows for Julia 1.9, 1.10, and latest 1.x.
- Optional Codecov upload is scaffolded; uncomment in `.github/workflows/ci.yml` and add `CODECOV_TOKEN` secret.

## Release & Registration

- CompatHelper: keeps `[compat]` up to date via PRs (`.github/workflows/CompatHelper.yml`).
- TagBot: creates GitHub releases/tags after registration (`.github/workflows/TagBot.yml`).
- Registration (General):
  - Ensure `Project.toml` has proper `[compat]` bounds and bump `version`.
  - Enable the JuliaRegistrator GitHub App on this repo, or use the web UI.
  - Trigger registration by commenting on a commit/PR: `@JuliaRegistrator register branch=main`.
  - After the General registry PR merges, TagBot will publish the GitHub release automatically.
