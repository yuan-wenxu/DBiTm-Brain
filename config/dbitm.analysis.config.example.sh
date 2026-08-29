#!/bin/bash
# Execution mode used
# local: run all steps locally
# hpc: submit all steps to HPC cluster using sbatch
execution_mode=hpc

host_dir=/path/to/host
methscan_dir=/path/to/methscan
# Chip size for methylation analysis. An integer value for the number of channels.s
chip_size=

# Shared input paths.
reference_fasta=/path/to/reference/genome.fa
chromhmm_cm=/path/to/ChromHMM.20220414.cm
cpg_reference=/path/to/cpg_nocontig.cr
gtf=/path/to/reference/genes.gtf.gz
# Optional headerless, one-gene-per-line file. Leave empty to skip marker plots.
marker_gene_file=
sbatch_log_dir=/path/to/sample/logs

# Site-analysis configuration.
site_min_sites=10
site_min_total_sites=100
# Leave empty to use chr1-chr19,chrX.
site_chromosomes=

# VMR-analysis output and filters. Filtered matrices are written below this root.
vmr_contexts="CA CC CG CT"
vmr_spot_filter_quantile=0.05
vmr_min_observed_spots=10
vmr_downstream_context=CG

# Slurm settings used only in HPC mode.
sbatch_job_name_prefix=dbitm_brain
sbatch_partition=
sbatch_requeue=false

sbatch_site_cpus=1
sbatch_site_mem=32G
sbatch_site_time=24:00:00

sbatch_vmr_cpus=1
sbatch_vmr_mem=32G
sbatch_vmr_time=24:00:00
