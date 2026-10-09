#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd -- "$(dirname -- "$0")/.."
r_env=default

if [[ ! -v CONDA_OVERRIDE_CUDA ]]; then
    pixi_system_info="$(pixi info)"
    if [[ "$pixi_system_info" != *"__cuda="* ]]; then
        export CONDA_OVERRIDE_CUDA=12.9
        echo "No CUDA driver detected; preparing the CUDA 12.9 environment for HPC."
    fi
fi

pixi install --environment "$r_env" --locked
for package in TxDb.Hsapiens.UCSC.hg19.knownGene GO.db; do
    if pixi run --environment "$r_env" --locked Rscript -e "quit(status=if (requireNamespace('$package', quietly=TRUE)) 0L else 1L)"; then
        echo "$package is already installed; skipping download."
    else
        pixi reinstall --environment "$r_env" "bioconductor-${package,,}" --locked --run-post-link-scripts
    fi
done

mkdir -p "${ROOT}/.vscode"

cat > "${ROOT}/.vscode/settings.json" <<EOF
{
    "r.rpath.linux": "\${workspaceFolder}/.pixi/envs/${r_env}/bin/R",
    "r.rterm.linux": "\${workspaceFolder}/.pixi/envs/${r_env}/bin/R",
    "r.env": {
        "RETICULATE_PYTHON": "\${workspaceFolder}/.pixi/envs/${r_env}/bin/python"
    }
}
EOF
