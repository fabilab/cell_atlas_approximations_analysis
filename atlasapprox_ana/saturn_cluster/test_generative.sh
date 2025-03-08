#!/bin/bash

# Activate the environment
#source /srv/scratch/fabilab/fabio/miniforge3/bin/activate && conda activate saturn

train_arg=""
if [ "x$1" == "x--train" ]; then
 train_args=" --train"
fi

# Set I/O folders
species="a_queenslandica"
guide_species="s_lacustris"
embeddings_fn="/srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/data/reference_atlases/esm1b_embeddings_summaries/${species}_gene_all_esm1b.pt"
model_fdn="/srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/data/reference_atlases/saturn_output_esm1b/output_nmacro1300_nhvg3500_epochs_p100_m30"

echo "Go to the right folder"
cd /srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/software/SATURN

echo "Run the generative model"
python generative.py \
 --in_data ${model_fdn}/in_csv.csv \
 --in_adata_path ${model_fdn}/saturn_results/final_adata.h5ad \
 --in_embeddings_path ${embeddings_fn} \
 --pretrain_model_path ${model_fdn}/pretrain_model.model \
 --metric_model_path ${model_fdn}/metric_model.model \
 --centroids_init_path ${model_fdn}/centroids.pkl \
 --gen_model_path ${model_fdn}/gen_test.model \
 --species ${species} \
 --guide_species ${guide_species} \
 --seed 42 \
 --epochs 500 \
 --work_dir ${model_fdn}/generative_${species}_guide_${guide_species}/ \
 ${train_args}
