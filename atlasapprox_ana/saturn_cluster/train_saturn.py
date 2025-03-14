"""Train (i.e. run) SATURN on the termites/fly data.


This must be run inside the Python 3.10 saturn conda environment: source /srv/scratch/fabilab/fabio/miniforge3/bin/activate && conda activate saturn

When run inside katana, this Python script is actually run by a shell script which already sets the environment. In theory, that should be preserved throughout subprocess.run

This comes roughly from here:

https://github.com/snap-stanford/SATURN/blob/main/Vignettes/frog_zebrafish_embryogenesis/Train%20SATURN.ipynb
"""
import os
import pathlib
import pandas as pd
import subprocess as sp
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train SATURN on the atlasapprox data.")
    parser.add_argument('--dry', action="store_true", help="Dry run")
    parser.add_argument('--skip-checks', action="store_true", help="Skip sanity checks")
    parser.add_argument('--n-macrogenes', default=2000, help="Number of macrogenes")
    parser.add_argument('--n-hvg', default=8000, help="Number of highly variable genes")
    parser.add_argument('--n-epochs', default=50, help="Number of epochs in metric learning")
    parser.add_argument('--n-pretrain-epochs', default=200, help="Number of epochs in pretraining")
    parser.add_argument('--leaveout', type=int, default=-1, help="Leave one species out (1-indexed) for testing")
    parser.add_argument('--protein-model', type=str, choices=['esm1b', 'esmc', 'esmc600'], default='esm1b', help='Which model to use')
    args = parser.parse_args()

    fasta_root_folder = pathlib.Path("/srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/data/reference_atlases/peptide_sequences/")
    h5ad_fdn = fasta_root_folder.parent / "h5ads"
    embeddings_summary_fdn = fasta_root_folder.parent / f"{args.protein_model}_embeddings_summaries/"
    output_fdn = fasta_root_folder.parent / f"saturn_output_{args.protein_model}" / f"output_nmacro{args.n_macrogenes}_nhvg{args.n_hvg}_epochs_p{args.n_pretrain_epochs}_m{args.n_epochs}"
    if args.leaveout >= 1:
        output_fdn = output_fdn.parent / (output_fdn.name + f"_leaveout_{args.leaveout}")
    os.makedirs(output_fdn, exist_ok=True)

    # Build the CSV used by SATURN to connect the species
    sample_dict = {}
    for h5ad_fn in h5ad_fdn.iterdir():
        species = h5ad_fn.stem
        print(species)
        embedding_summary_fn = embeddings_summary_fdn / f"{species}_gene_all_{args.protein_model}.pt"
        if not embedding_summary_fn.exists():
            print(" Embedding summary file not found, skipping")
            continue

        sample_dict[species] = {
            "path": h5ad_fn,
            "embedding_path": embedding_summary_fn,
        }
    df = pd.DataFrame(sample_dict).T.reset_index().rename(columns={"index": "species"})
    # NOTE: this is just to make sure they are already alphabetical to deal with hidden bugs in SATURN
    df = df.sort_values("species")

    # Leave one species out if requested, for testing/validation purposes
    if args.leaveout >= 1:
        idx = [x for x in range(len(df)) if x != args.leaveout - 1]
        df = df.iloc[idx]

    saturn_csv_fn = output_fdn / "in_csv.csv"
    if not args.dry:
        df.to_csv(saturn_csv_fn, index=False)

    # Sanity check: verify all features used in the h5ad var_names have a corresponding embedding
    if not args.skip_checks:
        for _, row in df.iterrows():
            print("Checking", row["species"])
            adata = __import__("anndata").read_h5ad(row["path"])
            embedding = __import__("torch").load(row["embedding_path"])
            features_h5ad = adata.var_names
            features_emb = pd.Index(embedding.keys())
            features_xor = set(features_h5ad) ^ set(features_emb)
            if len(features_xor) > 0:
                if len(features_xor) < 10:
                    nfea = len(features_xor)
                    print(f"Features in h5ad but not in embedding ({nfea}), correcting h5ad file:", features_xor)
                    adata = adata[:, features_emb]
                    adata.write_h5ad(row["path"])
                else:
                    assert features_h5ad.isin(features_emb).all()
            del adata, embedding
            __import__("gc").collect()

    # Run SATURN
    script_fn = pathlib.Path("/srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/software/SATURN/train-saturn.py")
    centroids_fn = output_fdn / "centroids.pkl"  # This is temp output to speed up later iterations (kmeans is costly, apparently)
    pretrain_model_fn = output_fdn / "pretrain_model.model"  # This is temp output to speed up later iterations (kmeans is costly, apparently)
    metric_model_fn = output_fdn / "metric_model.model"  # This is temp output to speed up later iterations (kmeans is costly, apparently)
    #scoring_maps_fn = output_fdn / "scoring_cell_type_maps.csv"  # This is temp output to speed up later iterations (kmeans is costly, apparently)

    call = [
        "python",
        str(script_fn),
        "--in_data",
        str(saturn_csv_fn),
        "--in_label_col=cellType",
        "--ref_label_col=cellType",
        f"--centroids_init_path={centroids_fn}",
        f"--work_dir={output_fdn}",  # This is general output
        "--seed=42",
        f"--epochs={args.n_epochs}",
        f"--pretrain_epochs={args.n_pretrain_epochs}",
        f"--num_macrogenes={args.n_macrogenes}",
        f"--hv_genes={args.n_hvg}",
        # Store model'
        f"--pretrain_model_path={pretrain_model_fn}",
        f"--metric_model_path={metric_model_fn}",
    ]
    print(" ".join(call))
    if not args.dry:
        sp.run(" ".join(call), shell=True, check=True)
