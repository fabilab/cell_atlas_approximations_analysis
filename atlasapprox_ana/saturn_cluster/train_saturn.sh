#!/bin/bash
#PBS -l walltime=18:00:00
#PBS -l ncpus=32
#PBS -l ngpus=1
#PBS -l mem=512GB
#PBS -l host=k097
#PBS -o /srv/scratch/fabilab/fabio/projects/atlasapprox_saturn/logs/train_output.log
#PBS -e /srv/scratch/fabilab/fabio/projects/atlasapprox_saturn/logs/train_error.log
#PBS -j oe

# NOTE: The fabilab GPU node has 64 cores, 2 GPUs with 48GB of VRAM each, and 1TB of memory.
# Because there is currently little virtualisation tooling to quota GPU VRAM to jobs, a job
# that requires a whole GPU should best request 32 CPUs and 512 GB of RAM even though that
# might not be strictly necessary. Or just request the whole machine by doubling that.

#export TF_CPP_MIN_LOG_LEVEL=2
#export TF_ENABLE_ONEDNN_OPTS=0

# Activate the environment
source /srv/scratch/fabilab/fabio/miniforge3/bin/activate
conda activate saturn

# Move to the source directory
cd /srv/scratch/fabilab/fabio/projects/atlasapprox_saturn/aasaturn/pipelines

# Create a log file name with the current date and time
LOG_FILE="/srv/scratch/fabilab/fabio/projects/atlasapprox_saturn/logs/train_output_$(date '+%Y-%m-%d_%H-%M-%S').log"

# Log the start time
echo "Job started at: $(date)" >> "$LOG_FILE"

# Run the pre-training code and capture runtime logs
python train_saturn.py --n-macro 1300 --n-hvg 3500 --n-epochs 30 --n-pretrain-epochs 100 \
  2>&1 | tee -a "$LOG_FILE"

# Log the end time
echo "Job ended at: $(date)" >> "$LOG_FILE"
