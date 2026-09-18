# Repository Guidelines

## Project Structure & Module Organization

The repository contains analysis scripts rather than an installable application. Python workflow stages live in `scripts-python/` and use numeric prefixes to show execution order. Equivalent or supporting Seurat workflows are in `scripts-r/`. Performance and data-summary utilities are kept in `benchmark/`. Dependencies and environments are defined by `pixi.toml` and the generated `pixi.lock`. Large sequencing inputs, H5AD files, plots, and analysis results should remain outside the repository.

## Build, Test, and Development Commands

Use Pixi for every Python and R command:

```bash
pixi install
pixi run python scripts-python/00-matrix-io-transcriptome.py --help
pixi run python scripts-python/02-matrix-qc-transcriptome.py input.h5ad
pixi run Rscript scripts-r/00-loadmRNA.R
```

There is no compilation step and no predefined Pixi task. Add dependencies with `pixi add <package>`; do not manually edit `pixi.lock` or install packages with system-level `pip`, Conda, or R.

## Coding Style & Naming Conventions

Use four-space indentation, type hints, `pathlib.Path`, descriptive snake_case names, and small single-purpose functions in Python. Keep command-line parsing in `parse_args()` and execution in `main()`. Follow existing tidyverse/Seurat conventions in R, using `<-` for assignment. Preserve numeric script prefixes and include the modality in filenames, such as `02-matrix-qc-methylation.py`. Avoid unrelated formatting changes.

## Visualization Conventions

Use this cluster palette in order: `#A73030`, `#E64B35`, `#2F5597`, `#4DBBD5`, `#CC79A7`, `#7E57C2`, `#C5A3E0`, `#D6A500`, `#FFE082`, `#2CA02C`, `#264653`, `#E7298A`, `#98DF8A`, and `#C49C94`. Assign colors deterministically by numeric cluster ID so that the same cluster has the same color across UMAP and spatial plots. Use `#4DBBD5` for single-color QC distributions and the `#E64B35`–white–`#4DBBD5` diverging scheme for spatial QC metrics when applicable.

## Testing Guidelines

No automated test framework or coverage threshold is currently configured. For significant changes, create a minimal synthetic dataset under `/tmp`, run only the affected workflow through `pixi run`, and verify matrix dimensions, required AnnData/Seurat fields, plots, and output paths. Run each modified CLI with `--help` and use `git diff --check` before submission. Remove all temporary test files after collecting results.

## Commit & Pull Request Guidelines

Recent history follows Conventional Commits: `feat(qc): ...`, `fix(io): ...`, or `refactor(vmr-analysis): ...`. Keep commits focused and use an imperative English summary. Pull requests should explain the scientific or pipeline change, list modified stages, provide exact validation commands, and note untested real-data cases. Include representative plot screenshots when visualization output changes, and link relevant issues or datasets without committing the data itself.
