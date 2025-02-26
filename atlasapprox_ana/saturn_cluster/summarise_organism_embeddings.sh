#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -l ncpus=4
#PBS -l mem=64GB
#PBS -o /srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/logs/summarise_embeddings_output.log
#PBS -e /srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/logs/summarise_embeddings_error.log
#PBS -j oe
#PBS -l ngpus=1
#PBS -P MWAC
#PBS -l host=k099
#y##PBS -P FABILAB
#y##PBS -l host=k097

# NOTE: The fabilab GPU node has 64 cores, 2 GPUs with 48GB of VRAM each, and 1TB of memory.
# Because there is currently little virtualisation tooling to quota GPU VRAM to jobs, a job
# that requires a whole GPU should best request 32 CPUs and 512 GB of RAM even though that
# might not be strictly necessary. Or just request the whole machine by doubling that.

# NOTE: SBF owns nodes k099, k100, k101. It seems like asking for one of those is key to get
# priority access to the GPUs. The other nodes (e.g. k106) have older GPUs (e.g. V100).

# Activate the environment
source /srv/scratch/fabilab/fabio/miniforge3/bin/activate
conda activate esmc

# Move to the source directory
cd /srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/atlasapprox_ana/saturn_cluster

# Create a log file name with the current date and time
LOG_FILE="/srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/logs/summarise_embeddings_output_$(date '+%Y-%m-%d_%H-%M-%S').log"

# Log the start time
echo "Job started at: $(date)" >> "$LOG_FILE"

# Run the pre-training code and capture runtime logs
python summarise_organism_embeddings.py --model esmc600 \
  2>&1 | tee -a "$LOG_FILE"

# Log the end time
echo "Job ended at: $(date)" >> "$LOG_FILE"
