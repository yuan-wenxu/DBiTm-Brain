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

for name in methscan_dir gtf chip_size; do
    require_value "$name"
done

require_dir "$methscan_dir" "MethSCAn directory"
require_file "$gtf" "GTF"
if [[ -n ${marker_gene_file:-} ]]; then
    require_file "$marker_gene_file" "marker gene file"
fi
if [[ ! "$chip_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: chip_size must be a positive integer." >&2
    exit 1
fi

pixi_project_dir=${pixi_project_dir:-$REPO_DIR}
if [[ ! -f "$pixi_project_dir/pixi.toml" ]]; then
    echo "Error: Pixi project not found: $pixi_project_dir/pixi.toml" >&2
    exit 1
fi
if ! command -v pixi >/dev/null 2>&1; then
    echo "Error: pixi executable not found." >&2
    exit 1
fi

if [[ ! "$vmr_min_observed_spots" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: vmr_min_observed_spots must be a positive integer." >&2
    exit 1
fi
if [[ ! "$vmr_downstream_context" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "Error: vmr_downstream_context contains unsupported characters." >&2
    exit 1
fi

run_pixi() {
    (
        cd "$pixi_project_dir"
        pixi run "$@"
    )
}

context_args=()
if [[ -n ${vmr_contexts:-} ]]; then
    read -r -a configured_contexts <<< "$vmr_contexts"
    if [[ ${#configured_contexts[@]} -eq 0 ]]; then
        echo "Error: vmr_contexts must contain at least one context when set." >&2
        exit 1
    fi
    downstream_selected=false
    for context in "${configured_contexts[@]}"; do
        if [[ "$context" == "$vmr_downstream_context" ]]; then
            downstream_selected=true
            break
        fi
    done
    if [[ "$downstream_selected" != true ]]; then
        echo "Error: vmr_contexts must include vmr_downstream_context ($vmr_downstream_context)." >&2
        exit 1
    fi
    context_args=(--contexts "${configured_contexts[@]}")
fi

vmr_output_dir=${methscan_dir%/*}/vmr_analysis
mkdir -p "$vmr_output_dir"

matrix_qc_args=(
    python -B "$REPO_DIR/scripts/vmr-analysis/01-matrix-qc.py"
    --methscan-dir "$methscan_dir"
    --output-dir "$vmr_output_dir"
    --spot-filter-quantile "$vmr_spot_filter_quantile"
    --min-vmr-observed-spots "$vmr_min_observed_spots"
)
matrix_qc_args+=("${context_args[@]}")

echo "[VMR 1/5] Matrix QC and filtering: $vmr_output_dir"
run_pixi "${matrix_qc_args[@]}"

filtered_root="$vmr_output_dir/filtered_methscan"
length_output="$vmr_output_dir/vmr_length_distribution"
mkdir -p "$length_output"

length_args=(
    python -B "$REPO_DIR/scripts/vmr-analysis/02-vmr-length.py"
    --methscan_dir "$filtered_root"
    --output-dir "$length_output"
)
length_args+=("${context_args[@]}")

echo "[VMR 2/5] Filtered VMR length distribution: $length_output"
run_pixi "${length_args[@]}"

context_matrix_dir="$filtered_root/$vmr_downstream_context/matrix"
fraction_matrix="$context_matrix_dir/methylation_fractions.csv.gz"
residual_matrix="$context_matrix_dir/mean_shrunken_residuals.csv.gz"
gene_table="$filtered_root/$vmr_downstream_context/vmr_to_genes.tsv"
require_file "$fraction_matrix" "filtered methylation-fraction matrix"
require_file "$residual_matrix" "filtered residual matrix"

echo "[VMR 3/5] Link $vmr_downstream_context VMRs to genes: $gene_table"
run_pixi python -B "$REPO_DIR/scripts/vmr-analysis/03-link-vmr-genes.py" \
    --gtf "$gtf" \
    --matrix "$fraction_matrix" \
    --context "$vmr_downstream_context" \
    --output "$gene_table"

if [[ -n ${marker_gene_file:-} ]]; then
    marker_output="$filtered_root/$vmr_downstream_context/marker_vmr_visualization"
    mkdir -p "$marker_output"
    echo "[VMR 4/5] Marker VMR heatmaps: $marker_output"
    run_pixi python -B "$REPO_DIR/scripts/vmr-analysis/04-marker-meth-visualization.py" \
        --matrix "$fraction_matrix" \
        --gene-table "$gene_table" \
        --gene-file "$marker_gene_file" \
        --chip-size "$chip_size" \
        --output-dir "$marker_output" \
        --row-suffix ".$vmr_downstream_context"
else
    echo "[VMR 4/5] Marker VMR heatmaps skipped: marker_gene_file is empty"
fi

cluster_output="$filtered_root/$vmr_downstream_context/residual_clustering"
mkdir -p "$cluster_output"
echo "[VMR 5/5] Residual clustering: $cluster_output"
run_pixi python -B "$REPO_DIR/scripts/vmr-analysis/05-cluster-residuals.py" \
    --matrix "$residual_matrix" \
    --output-dir "$cluster_output" \
    --chip-size "$chip_size" \
    --row-suffix ".$vmr_downstream_context"

echo "VMR analysis complete: $vmr_output_dir"
