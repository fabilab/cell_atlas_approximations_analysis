#!/bin/bash

# Activate the environment
#source /srv/scratch/fabilab/fabio/miniforge3/bin/activate && conda activate saturn

# Set I/O folders
model_fdn="/srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/data/reference_atlases/saturn_output_esm1b/output_nmacro1300_nhvg3500_epochs_p100_m30"

if [ "$#" -ne 1 ]; then
  echo "Positional argument needed: species."
  exit 2
fi

species=$1
case $species in
  t_cynocephalus | thylacine)
    species="t_cynocephalus"
    guide_species="m_musculus"
    embeddings_fn="/srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/data/noatlas_species/esm1b_embeddings_summaries/${species}_gene_all_esm1b.pt"
    ;;
  c_punctulatus | cpun)
    species="cpun"
    guide_species="d_melanogaster"
    embeddings_fn="/srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/data/noannotation_species/esm1b_embeddings_summaries/Cmer_gene_all_esm1b.pt"
    ;;
  *)
   echo "Species not recognised."
   exit 3
   ;;
esac

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
 --epochs 200 \
 --work_dir ${model_fdn}/generative_${species}_guide_${guide_species}/ \
 --train
