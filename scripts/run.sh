#!/bin/bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)
START_DIR=$(pwd -P)
PROGRAM_NAME=$(basename "$0")

show_help() {
    cat <<EOF
Usage: $PROGRAM_NAME <site|vmr|all> [--config FILE]

Steps:
  site          Run site-level methylation analyses
  vmr           Run VMR matrix QC and downstream analyses
  all           Run or submit both independent analyses

Options:
  --config FILE Analysis configuration (default: ./dbitm.analysis.config.sh)
  -h, --help    Show this help message and exit
EOF
}

require_option_value() {
    if [[ $# -lt 2 || ${2:-} == --* ]]; then
        echo "Error: option '$1' requires a value." >&2
        exit 1
    fi
}

if [[ $# -eq 0 || ${1:-} == -h || ${1:-} == --help ]]; then
    show_help
    exit 0
fi

selection=$1
shift
case "$selection" in
    site|vmr|all) ;;
    *)
        echo "Error: unsupported step '$selection'. Valid steps: site, vmr, all." >&2
        exit 1
        ;;
esac

config_file=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            require_option_value "$@"
            config_file=$2
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: unknown option or argument '$1'." >&2
            exit 1
            ;;
    esac
done

if [[ -z "$config_file" ]]; then
    config_file="$START_DIR/dbitm.analysis.config.sh"
fi
if [[ ! -f "$config_file" ]]; then
    echo "Error: config file not found: $config_file" >&2
    echo "Copy config/dbitm.analysis.config.example.sh and edit the paths first." >&2
    exit 1
fi
config_file=$(realpath "$config_file")

# shellcheck source=/dev/null
source "$config_file"

mode=${execution_mode}
case "$mode" in
    local|hpc) ;;
    *)
        echo "Error: execution mode must be local or hpc." >&2
        exit 1
        ;;
esac

if [[ "$selection" == all ]]; then
    selected_steps=(site vmr)
else
    selected_steps=("$selection")
fi

for step in "${selected_steps[@]}"; do
    worker="$SCRIPT_DIR/$step.sh"
    if [[ ! -f "$worker" ]]; then
        echo "Error: worker script not found: $worker" >&2
        exit 1
    fi
done

echo "Using config: $config_file"
echo "Execution mode: $mode"

if [[ "$mode" == local ]]; then
    for step in "${selected_steps[@]}"; do
        echo "Running $step locally"
        bash "$SCRIPT_DIR/$step.sh" "$config_file"
    done
    exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "Error: sbatch executable not found." >&2
    exit 1
fi
for name in sbatch_job_name_prefix sbatch_log_dir; do
    if [[ -z ${!name:-} ]]; then
        echo "Error: $name must be set in HPC mode." >&2
        exit 1
    fi
done

validate_resources() {
    local step=$1
    local cpus_name="sbatch_${step}_cpus"
    local mem_name="sbatch_${step}_mem"
    local time_name="sbatch_${step}_time"
    local cpus=${!cpus_name:-}
    local memory=${!mem_name:-}
    local walltime=${!time_name:-}

    if [[ ! "$cpus" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: $cpus_name must be a positive integer." >&2
        exit 1
    fi
    if [[ -z "$memory" || -z "$walltime" ]]; then
        echo "Error: $mem_name and $time_name must be set in HPC mode." >&2
        exit 1
    fi
}

for step in "${selected_steps[@]}"; do
    validate_resources "$step"
done
mkdir -p "$sbatch_log_dir"

case "${sbatch_requeue:-false}" in
    True|true|TRUE|1|yes|YES) requeue=true ;;
    False|false|FALSE|0|no|NO|"") requeue=false ;;
    *)
        echo "Error: sbatch_requeue must be true or false." >&2
        exit 1
        ;;
esac

submit_step() {
    local step=$1
    local cpus_name="sbatch_${step}_cpus"
    local mem_name="sbatch_${step}_mem"
    local time_name="sbatch_${step}_time"
    local worker="$SCRIPT_DIR/$step.sh"
    local wrapped_command
    local job_id
    local sbatch_args=(
        --job-name="${sbatch_job_name_prefix}_${step}"
        --cpus-per-task="${!cpus_name}"
        --mem="${!mem_name}"
        --time="${!time_name}"
        --output="$sbatch_log_dir/%x.%j.out"
        --error="$sbatch_log_dir/%x.%j.err"
    )

    [[ -n ${sbatch_partition:-} ]] && sbatch_args+=(--partition="$sbatch_partition")
    [[ "$requeue" == true ]] && sbatch_args+=(--requeue)

    printf -v wrapped_command 'exec bash %q %q' "$worker" "$config_file"
    job_id=$(sbatch "${sbatch_args[@]}" --wrap="$wrapped_command")
    echo "Submitted $step: $job_id"
}

for step in "${selected_steps[@]}"; do
    submit_step "$step"
done
