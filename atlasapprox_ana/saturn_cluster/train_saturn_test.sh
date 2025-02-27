#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -l mem=32GB
#PBS -l host=k097
#PBS -o /srv/scratch/fabilab/fabio/projects/atlasapprox_saturn/logs/train_output.log
#PBS -e /srv/scratch/fabilab/fabio/projects/atlasapprox_saturn/logs/train_error.log
#PBS -j oe

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
echo "Job (test) started at: $(date)" >> "$LOG_FILE"

# Run the pre-training code and capture runtime logs
python train_saturn.py --n-macro 10 --n-hvg 20 --n-epochs 1 --n-pretrain-epochs 1 \
  2>&1 | tee -a "$LOG_FILE"

# Log the end time
echo "Job (test) ended at: $(date)" >> "$LOG_FILE"
