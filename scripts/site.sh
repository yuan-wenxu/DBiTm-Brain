#!/bin/bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

if [[ $# -ne 1 ]]; then
    echo "Usage: $(basename "$0") <analysis-config.sh>" >&2
    exit 1
fi

config_file=$1
if [[ ! -f "$config_file" ]]; then
    echo "Error: config file not found: $config_file" >&2
    exit 1
fi
config_file=$(realpath "$config_file")

# shellcheck source=/dev/null
source "$config_file"

require_value() {
    local name=$1
    if [[ -z ${!name:-} ]]; then
        echo "Error: $name must be set in $config_file" >&2
        exit 1
    fi
}

require_file() {
    local path=$1
    local label=$2
    if [[ ! -f "$path" ]]; then
        echo "Error: $label not found: $path" >&2
        exit 1
    fi
}

require_dir() {
    local path=$1
    local label=$2
    if [[ ! -d "$path" ]]; then
        echo "Error: $label not found: $path" >&2
        exit 1
    fi
}

for name in host_dir reference_fasta chromhmm_cm cpg_reference; do
    require_value "$name"
done

require_dir "$host_dir" "host coverage directory"
require_file "$reference_fasta" "reference FASTA"
require_file "$chromhmm_cm" "ChromHMM mask"
require_file "$cpg_reference" "CpG reference"

site_output_dir=${host_dir%/*/*}/site_analysis
mkdir -p "$site_output_dir"

pixi_project_dir=${pixi_project_dir:-$REPO_DIR}
if [[ ! -f "$pixi_project_dir/pixi.toml" ]]; then
    echo "Error: Pixi project not found: $pixi_project_dir/pixi.toml" >&2
    exit 1
fi
if ! command -v pixi >/dev/null 2>&1; then
    echo "Error: pixi executable not found." >&2
    exit 1
fi

if [[ ! "$site_min_sites" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: site_min_sites must be a positive integer." >&2
    exit 1
fi
if [[ ! "$site_min_total_sites" =~ ^[0-9]+$ ]]; then
    echo "Error: site_min_total_sites must be a non-negative integer." >&2
    exit 1
fi

run_pixi() {
    (
        cd "$pixi_project_dir"
        pixi run "$@"
    )
}

chromhmm_output="$site_output_dir/01.chromhmm_methylation"
context_output="$site_output_dir/02.context_coverage"
mkdir -p "$chromhmm_output" "$context_output"

echo "[site 1/2] ChromHMM region methylation: $chromhmm_output"
run_pixi python -B "$REPO_DIR/scripts/site-analysis/01-region-methylation.py" \
    --host-dir "$host_dir" \
    --chromhmm-cm "$chromhmm_cm" \
    --cpg-reference "$cpg_reference" \
    --output-dir "$chromhmm_output" \
    --min-sites "$site_min_sites" \
    --min-total-sites "$site_min_total_sites"

context_args=(
    python -B "$REPO_DIR/scripts/site-analysis/02-context-coverage.py"
    --host-dir "$host_dir"
    --reference-fasta "$reference_fasta"
    --output-dir "$context_output"
    --min-total-sites "$site_min_total_sites"
)
if [[ -n ${site_chromosomes:-} ]]; then
    context_args+=(--chromosomes "$site_chromosomes")
fi

echo "[site 2/2] Context coverage: $context_output"
run_pixi "${context_args[@]}"

echo "Site analysis complete: $site_output_dir"
