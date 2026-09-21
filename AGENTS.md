# Repository Guidelines

## Project Structure & Module Organization

Primary workflows live in `scripts/`, reusable utilities in `tools/`, and standalone analyses in `benchmark/`. Preserve numeric prefixes on ordered workflows and include the modality in filenames when relevant. Keep large data files, images, plots, and generated results outside Git.

## Environment & Dependency Management

Work in the Linux container with Bash and Linux-style paths. For Windows-mounted data under `/mnt/...`, account for slower I/O, permissions, filename case, and line-ending differences.

Pixi is the supported environment and dependency manager. Inspect `pixi.toml` and its tasks before running commands:

```bash
pixi install
pixi run python path/to/script.py --help
pixi add <package>
```

Run project tools through `pixi run`. Do not use system-level package managers or runtimes, and never edit `pixi.lock` manually. Do not assume `test` or `lint` tasks exist.

## Coding Style & Naming Conventions

Use four-space indentation, type hints, `pathlib.Path`, descriptive `snake_case` names, and uppercase module constants. Keep CLI parsing in `parse_args()` and execution in `main()`. Prefer small, single-purpose functions and avoid unrelated formatting or refactoring. No formatter or linter is currently configured.

## Visualization Conventions

Use this cluster palette in order: `#A73030`, `#E64B35`, `#2F5597`, `#4DBBD5`, `#CC79A7`, `#7E57C2`, `#C5A3E0`, `#D6A500`, `#FFE082`, `#2CA02C`, `#264653`, `#E7298A`, `#98DF8A`, `#C49C94`. Assign colors deterministically by numeric cluster ID so UMAP and spatial plots remain consistent. Use `#4DBBD5` for single-color QC distributions and `#E64B35`–white–`#4DBBD5` for diverging spatial QC metrics.

## Testing Guidelines

There is no automated test suite or coverage requirement. For significant changes, create minimal synthetic data under `/tmp` and run only the affected workflow through `pixi run`. Verify dimensions, required fields, output paths, and plots as applicable. Run modified CLIs with `--help` and `git diff --check`. Remove temporary artifacts afterward.

## Git, Commits & Pull Requests

Check `git status` and relevant diffs before editing. Preserve unrelated uncommitted work. Do not commit, push, rebase, rewrite history, delete branches, or run destructive Git commands unless explicitly requested.

Write commit messages in English using Conventional Commits: `<type>(<scope>): <imperative summary>`, for example `fix(integration): align spot pair column names`. Common types are `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, and `style`. Keep each commit focused. Pull requests should explain the scientific or pipeline change, list affected files or stages, give exact validation commands, note untested real-data cases, link relevant issues or datasets, and include representative screenshots for visualization changes.
