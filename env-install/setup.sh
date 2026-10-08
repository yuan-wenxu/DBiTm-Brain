#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd -- "$(dirname -- "$0")/.."
r_env=default

pixi install --locked
for package in TxDb.Hsapiens.UCSC.hg19.knownGene GO.db; do
    if pixi run Rscript -e "quit(status=if (requireNamespace('$package', quietly=TRUE)) 0L else 1L)"; then
        echo "$package is already installed; skipping download."
    else
        pixi reinstall "bioconductor-${package,,}" --locked --run-post-link-scripts
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
