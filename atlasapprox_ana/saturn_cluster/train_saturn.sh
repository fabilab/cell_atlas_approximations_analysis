#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l ncpus=8
#PBS -l ngpus=1
#PBS -l mem=256GB
#PBS -o /srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/logs/train_output.log
#PBS -e /srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/logs/train_error.log
#PBS -j oe
#PBS -P MWAC
#PBS -l host=k099
#y#PBS -P FABILAB
#y#PBS -l host=k097

# NOTE: The fabilab GPU node has 64 cores, 2 GPUs with 48GB of VRAM each, and 1TB of memory.
# Because there is currently little virtualisation tooling to quota GPU VRAM to jobs, a job
# that requires a whole GPU should best request 32 CPUs and 512 GB of RAM even though that
# might not be strictly necessary. Or just request the whole machine by doubling that.

# NOTE: k099 from SBF has 32 cores, 1 TB of memory, and 4 GPUs, so if we want 1 GPU at a time
# we can request one fourth of everything.

#export TF_CPP_MIN_LOG_LEVEL=2
#export TF_ENABLE_ONEDNN_OPTS=0

# Activate the environment
source /srv/scratch/fabilab/fabio/miniforge3/bin/activate
conda activate saturn

# Move to the source directory
cd /srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/atlasapprox_ana/saturn_cluster

# Create a log file name with the current date and time
LOG_FILE="/srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/logs/train_output_$(date '+%Y-%m-%d_%H-%M-%S').log"

# Log the start time
echo "Job started at: $(date)" >> "$LOG_FILE"

# Run the pre-training code and capture runtime logs
python train_saturn.py --n-macro 2000 --n-hvg 8000 --n-epochs 50 --n-pretrain-epochs 100 --protein-model esm1b \
  2>&1 | tee -a "$LOG_FILE"

# Log the end time
echo "Job ended at: $(date)" >> "$LOG_FILE"
