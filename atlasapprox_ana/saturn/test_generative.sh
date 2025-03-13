#!/bin/bash
EMBDEDDING_SUMMARY_FDN=/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/esm_embeddings_summaries
SATURN_OUTPUT_FDN=/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/output_nmacro6_nhvg13_epochs_p1_m1
SPECIES=a_queenslandica
GUIDE=s_lacustris
python generative.py \
  --species ${SPECIES} \
  --guide ${GUIDE} \
  --org fabilab \
  --seed 42 \
  --in_data ${SATURN_OUTPUT_FDN}/in_csv.csv \
  --in_adata_path ${SATURN_OUTPUT_FDN}/saturn_results/final_adata.h5ad \
  --in_embeddings_path ${EMBDEDDING_SUMMARY_FDN}/${SPECIES}_gene_all_esm1b.pt \
  --centroids_init_path ${SATURN_OUTPUT_FDN}/centroids.pkl \
  --pretrain_model_path ${SATURN_OUTPUT_FDN}/pretrain_model.model \
  --metric_model_path ${SATURN_OUTPUT_FDN}/metric_model.model \
  --gen_model_path ${SATURN_OUTPUT_FDN}/gen_model.model
