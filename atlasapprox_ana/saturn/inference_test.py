"""Test inference with SATURN-like model on the termites/fly data.

This must be run inside the Python 3.10 saturn conda environment: source ~/miniconda3/bin/activate && conda activate saturn

"""

import os
import pathlib
import pandas as pd
import subprocess as sp
import argparse


leaveout_list = [
    "a_queenslandica",
    "a_thaliana",
    "c_elegans",
    "c_gigas",
    "c_hemisphaerica",
    "d_melanogaster",
    "d_rerio",
    "h_miamia",
    "h_sapiens",
    "i_pulchra",
    "l_minuta",
    "m_leidyi",
    "m_murinus",
    "m_musculus",
    "n_vectensis",
    "p_crozieri",
    "p_dumerilii",
    "s_lacustris",
    "s_mansoni",
    "s_mediterranea",
    "s_pistillata",
    "t_adhaerens",
    "t_aestivum",
    "x_laevis",
    "z_mays",
]
leaveout_dict = {key: i + 1 for i, key in enumerate(leaveout_list)}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train SATURN on the atlasapprox data."
    )
    parser.add_argument("--dry", action="store_true", help="Dry run")
    parser.add_argument(
        "--n-macrogenes",
        default=6,
        help="Number of macrogenes (only used to find the model)",
    )
    parser.add_argument("--n-hvg", default=13, help="Number of highly variable genes")
    parser.add_argument(
        "--n-epochs", default=1, help="Number of epochs in metric learning"
    )
    parser.add_argument(
        "--n-pretrain-epochs", default=1, help="Number of epochs in pretraining"
    )
    parser.add_argument(
        "--species",
        type=str,
        help="Infer embedding for data of this species",
        default=None,
    )
    parser.add_argument(
        "--train", action="store_true", help="Whether to train the transfer model"
    )
    parser.add_argument(
        "--leaveout",
        type=str,
        default=None,
        help="Use leaveout-traned model without this species for inference.",
    )
    parser.add_argument(
        "--secondary-analysis", action="store_true", help="Perform secondary analysis"
    )
    parser.add_argument(
        "--random-weights",
        action="store_true",
        help="Initialise the top layer with random weights.",
    )
    parser.add_argument(
        "--encoder",
        choices=["pretrain", "metric"],
        default="metric",
        help="Which encoder to use",
    )
    parser.add_argument(
        "--n-finetune-epochs", default=10, help="Number of epochs in finetuning"
    )
    parser.add_argument(
        "--guide-species", type=str, default=None, help="Guide species for inference"
    )
    args = parser.parse_args()

    if args.leaveout is not None and args.species is None:
        args.species = args.leaveout

    fasta_root_folder = pathlib.Path(
        "/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/peptide_sequences/"
    )
    embedding_root_fdn = fasta_root_folder.parent / "esm_embeddings"
    embeddings_summary_fdn = embedding_root_fdn.parent / "esm_embeddings_summaries/"
    h5ad_fdn = embeddings_summary_fdn.parent / "h5ads"
    training_output_fdn = (
        embeddings_summary_fdn.parent
        / f"output_nmacro{args.n_macrogenes}_nhvg{args.n_hvg}_epochs_p{args.n_pretrain_epochs}_m{args.n_epochs}"
    )
    if args.leaveout is not None:
        training_output_fdn = (
            training_output_fdn.parent
            / f"{training_output_fdn.stem}_leaveout_{args.leaveout}"
        )
    # There is a "from_kdm" folder that contains the big models, just to make sure I don't delete them by mistake
    if not training_output_fdn.exists():
        training_output_fdn = (
            embeddings_summary_fdn.parent
            / "from_kdm"
            / f"output_nmacro{args.n_macrogenes}_nhvg{args.n_hvg}_epochs_p{args.n_pretrain_epochs}_m{args.n_epochs}"
        )
        if args.leaveout is not None:
            leaveout_n = leaveout_dict[args.leaveout]
            training_output_fdn = (
                training_output_fdn.parent
                / f"{training_output_fdn.stem}_leaveout_{leaveout_n}"
            )
    centroids_fn = (
        training_output_fdn / "centroids.pkl"
    )  # This is temp output to speed up later iterations (kmeans is costly, apparently)
    pretrain_model_fn = training_output_fdn / "pretrain_model.model"
    metric_model_fn = training_output_fdn / "metric_model.model"
    output_fdn = training_output_fdn / "inference_output"
    training_csv_fn = training_output_fdn / "in_csv.csv"
    trained_adata_path = training_output_fdn / "saturn_results" / "final_adata.h5ad"

    # Build the CSV used by SATURN to connect the species
    for h5ad_fn in h5ad_fdn.iterdir():
        species = h5ad_fn.stem
        if species != args.species:
            continue
        print(species)
        embedding_summary_fn = embeddings_summary_fdn / f"{species}_gene_all_esm1b.pt"
        if not embedding_summary_fn.exists():
            print(" Embedding summary file not found, skipping")
            continue
        break
    else:
        raise IOError("adata or embedding summary files not found")

    # Sanity check: verify all features used in the h5ad var_names have a corresponding embedding
    row = {"species": species, "path": h5ad_fn, "embedding_path": embedding_summary_fn}
    print("Checking", row["species"])
    adata = __import__("anndata").read_h5ad(row["path"])
    embedding = __import__("torch").load(row["embedding_path"])
    features_h5ad = adata.var_names
    features_emb = pd.Index(embedding.keys())
    features_xor = set(features_h5ad) ^ set(features_emb)
    if len(features_xor) > 0:
        if len(features_xor) < 10:
            nfea = len(features_xor)
            print(
                f"Features in h5ad but not in embedding ({nfea}), correcting h5ad file:",
                features_xor,
            )
            adata = adata[:, features_emb]
            adata.write_h5ad(row["path"])
        else:
            assert features_h5ad.isin(features_emb).all()
    del adata, embedding
    __import__("gc").collect()

    # Run SATURN inference
    script_fn = (
        pathlib.Path("/home/fabio/projects/termites")
        / "software"
        / "SATURN"
        / "inference.py"
    )
    call = [
        "python",
        str(script_fn),
        f"--in_adata_path={h5ad_fn}",
        f"--in_embeddings_path={embedding_summary_fn}",
        "--in_label_col=cellType",
        "--ref_label_col=cellType",
        f"--centroids_init_path={centroids_fn}",
        f"--pretrain_model_path={pretrain_model_fn}",
        f"--metric_model_path={metric_model_fn}",
        f"--work_dir={output_fdn}",  # This is general output
        f"--hv_genes={args.n_hvg}",
        "--seed=42",
        f"--species={args.species}-inf",
        f"--training_csv_path={training_csv_fn}",
        f"--encoder={args.encoder}",
    ]
    if args.train:
        call += [
            "--train",
            f"--trained_adata_path={trained_adata_path}",
            f"--epochs={args.n_finetune_epochs}",
        ]
    if args.random_weights:
        call += [
            "--random_weights",
        ]
    if args.guide_species is not None:
        call += [
            f"--guide_species={args.guide_species}",
        ]

    print(" ".join(call))
    if not args.dry:
        sp.run(" ".join(call), shell=True, check=True)

        if args.secondary_analysis:
            print("Begin secondary analysis")
            import numpy as np
            import anndata
            import scanpy as sc
            import matplotlib.pyplot as plt
            import seaborn as sns

            for result_h5ad in output_fdn.iterdir():
                if args.species not in result_h5ad.stem:
                    continue
                if f"guide_{args.guide_species}" not in result_h5ad.stem:
                    continue
                if args.train and str(result_h5ad).endswith("finetuned.h5ad"):
                    break
                elif (not args.train) and str(result_h5ad).endswith("zeroshot.h5ad"):
                    break
            else:
                raise IOError("Inference output h5ad file not found")

            print("Load h5ad for inference and training")
            adata_inf = anndata.read_h5ad(result_h5ad)
            adata_train = anndata.read_h5ad(trained_adata_path)

            print("Limit training data to the guide/closest species")
            adata_train = adata_train[
                adata_train.obs["species"] == adata_inf.uns["guide_species"]
            ]
            adata = anndata.concat([adata_inf, adata_train])

            print("Now we can make obs unique")
            adata.obs_names_make_unique()

            print("PCA")
            sc.pp.pca(adata)

            print("KNN")
            sc.pp.neighbors(adata)

            print("UMAP")
            sc.tl.umap(adata, n_components=2)

            print("Standardise some cell type names and set new column")

            def mapfun(ct):
                return {
                    "filament": "filamentous",
                    "glia": "glial",
                    "parenchyma": "parenchymal",
                }.get(ct, ct)

            adata.obs["cell_type"] = pd.Categorical(
                adata.obs["ref_labels"].astype(str).map(mapfun)
            )

            print("Visualise")
            plt.ion()
            plt.close("all")

            fig3, axs = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
            sc.pl.umap(
                adata,
                color="species",
                title="Species",
                add_outline=True,
                size=20,
                ax=axs[0],
            )
            axs[0].legend(ncol=1, fontsize=6, loc="best")
            for i, species in enumerate(
                [adata_inf.uns["guide_species"], f"{args.species}-inf"]
            ):
                ax = axs[i + 1]
                adata.obs["cell_type_tmp"] = adata.obs["cell_type"].astype(str)
                adata.obs.loc[adata.obs["species"] != species, "cell_type_tmp"] = (
                    "other species"
                )
                cell_types = list(sorted(adata.obs["cell_type_tmp"].unique()))
                cell_types.remove("other species")
                colors = sns.color_palette("husl", n_colors=len(cell_types))
                palette = dict(zip(cell_types, colors))
                palette["other species"] = (0.5, 0.5, 0.5)
                sc.pl.umap(
                    adata,
                    color="cell_type_tmp",
                    title=f"Cell Type ({species})",
                    add_outline=True,
                    size=15,
                    palette=palette,
                    ax=ax,
                )
                ax.legend(ncol=2, fontsize=6, loc="best")
            for ax in axs:
                ax.set(xlabel=None, ylabel=None, xticks=[], yticks=[])
            fig3.tight_layout()

            print("Closest cell type matches")
            print("10 absolute closest")
            import torch

            cdis = torch.cdist(
                torch.tensor(adata_inf.X).to("cuda"),
                torch.tensor(adata_train.X).to("cuda"),
            )
            closest = torch.topk(cdis.ravel(), 10, largest=False)
            close_idx = np.unravel_index(closest.indices.cpu().numpy(), cdis.shape)
            for i_inf, i_train in zip(*close_idx):
                print(
                    f"{adata_inf.obs['ref_labels'].values[i_inf]} -> {adata_train.obs['ref_labels'].values[i_train]}"
                )

            print("10 absolute closest in UMAP space")
            import torch

            umap_inf = adata[adata.obs["species"] == f"{args.species}-inf"].obsm[
                "X_umap"
            ]
            umap_train = adata[adata.obs["species"] != f"{args.species}-inf"].obsm[
                "X_umap"
            ]
            cdis = torch.cdist(
                torch.tensor(umap_inf).to("cuda"),
                torch.tensor(umap_train).to("cuda"),
            )
            closest_umap = torch.topk(cdis.ravel(), 10, largest=False)
            close_idx = np.unravel_index(closest_umap.indices.cpu().numpy(), cdis.shape)
            for i_inf, i_train in zip(*close_idx):
                print(
                    f"{adata_inf.obs['ref_labels'].values[i_inf]} -> {adata_train.obs['ref_labels'].values[i_train]}"
                )

            print("By cell type")
            from collections import Counter

            closest = cdis.min(axis=1)
            close_dic = Counter()
            for i_inf, i_train in enumerate(closest.indices.cpu().numpy()):
                close_dic[
                    (
                        adata_inf.obs["ref_labels"].values[i_inf],
                        adata_train.obs["ref_labels"].values[i_train],
                    )
                ] += 1
            close_dic = pd.Series(close_dic)
            close_dic = close_dic.unstack(fill_value=0)
            close_dic_frac = (1.0 * close_dic.T / close_dic.sum(axis=1)).T
            for idx, row in close_dic.iterrows():
                print(idx)
                for ct, count in row.nlargest(3).items():
                    if count == 0:
                        break
                    pct = int((100 * close_dic_frac.loc[idx, ct]).round(0))
                    print(f"  {ct}: {count} ({pct}%)")

            print(
                "Get percentages of correct predictions, whenever the correct cell type is found in the guide species"
            )
            # NOTE: this is somewhat unfair towards rare cell types because they can fall into the attraction basin of much bigger types
            # TODO: set up a statistically fairER test below
            from collections import defaultdict

            cell_types_guide = adata_train.obs["ref_labels"].cat.categories
            score = defaultdict(list)
            for idx, row in close_dic.iterrows():
                if idx not in cell_types_guide:
                    score["missing_from_guide"].append(idx)
                else:
                    best_match = row.idxmax()
                    if best_match == idx:
                        score["correct"].append(idx)
                    else:
                        score["incorrect"].append((idx, best_match))
            print(score)
            print(
                f"Correct: {len(score['correct'])}, incorrect: {len(score['incorrect'])}, missing from guide: {len(score['missing_from_guide'])}"
            )
