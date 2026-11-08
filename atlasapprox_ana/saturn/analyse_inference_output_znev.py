"""Analyse the output of SATURN

This script requires anndata and scanpy. One way to do that is to use the SATURN conda environment:

source ~/miniconda3/bin/activate && conda activate saturn

"""

import os
import sys
import pathlib
import numpy as np
import pandas as pd
import torch
import anndata
import scanpy as sc
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

full_name_dict = {
    "a_queenslandica": "Amphimedon queenslandica",
    "h_sapiens": "Homo sapiens",
    "m_musculus": "Mus musculus",
    "m_murinus": "Microcebus murinus",
    "d_rerio": "Danio rerio",
    "x_laevis": "Xenopus laevis",
    "t_adhaerens": "Trichoplax adhaerens",
    "s_lacustris": "Spongilla lacustris",
    "d_melanogaster": "Drosophila melanogaster",
    "l_minuta": "Lemna minuts",
    "a_thaliana": "Arabidopsis thaliana",
    "z_mays": "Zea mays",
    "f_vesca": "Fragaria vesca",
    "o_sativa": "Oryza sativa",
    "c_elegans": "Caenorhabditis elegans",
    "s_purpuratus": "Strongylocentrotus purpuratus",
    "s_pistillata": "Stylophora pistillata",
    "i_pulchra": "Isodiametra pulchra",
    "c_gigas": "Crassostrea gigas",
    "c_hemisphaerica": "Clytia hemisphaerica",
    "h_miamia": "Hofstenia miamia",
    "m_leidyi": "Mnemiopsis leidyi",
    "n_vectensis": "Nematostella vectensis",
    "p_crozieri": "Pseudoceros crozieri",
    "p_dumerilii": "Platynereis dumerilii",
    "s_mansoni": "Schistosoma mansoni",
    "s_mediterranea": "Schmidtea mediterranea",
    "t_aestivum": "Triticum aestivum",
}
termite_dict = {
    "dmel": ["d_melanogaster.h5ad", "d_melanogaster_gene_all_esm1b.pt"],
    "znev": ["znev_prepared_for_SATURN-xfer.h5ad", "Znev_gene_all_esm1b.pt"],
}
termite_full_names_dict = {
    "znev": "Zootermopsis nevadensis",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse the output of SATURN-xfer onto Zootermopsis termite."
    )
    parser.add_argument(
        "--n-macrogenes",
        default=2000,
        help="Number of macrogenes (only used to find the model)",
    )
    parser.add_argument(
        "--n-hvg",
        default=8000,
        help="Number of highly variable genes (only used to find the model)",
    )
    parser.add_argument(
        "--n-epochs",
        default=50,
        help="Number of epochs in metric learning (only used to find the model)",
    )
    parser.add_argument(
        "--n-pretrain-epochs",
        default=100,
        help="Number of epochs in pretraining (only used to find the model)",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="znev",
        help="Species that was inferred",
    )
    parser.add_argument(
        "--basemodel-kind",
        default="final",
        type=str,
        help="Which SATURN trained model was used as base for the transfer learning: pretrain, final, or the epoch number of the metric learning (e.g. 50)",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=10,
        help="How many epochs of fine tuning (used to find the SATURN-xfer output files)",
    )
    parser.add_argument(
        "--savefig",
        action="store_true",
        help="Whether to save the figures",
    )
    args = parser.parse_args()

    output_fdn = pathlib.Path(
        f"/home/fabio/projects/compressed_atlas/cell_atlas_approximations_analysis/data/reference_atlases/saturn_outputs_full_runs/output_nmacro{args.n_macrogenes}_nhvg{args.n_hvg}_epochs_p{args.n_pretrain_epochs}_m{args.n_epochs}"
    )
    if not output_fdn.exists():
        sys.exit(f"Output folder {output_fdn} does not exist")

    saturn_csv_fn = output_fdn / "in_csv.csv"
    df = pd.read_csv(saturn_csv_fn, index_col="species")

    if args.basemodel_kind == "pretrain":
        saturn_h5ad = output_fdn / "saturn_results" / "adata_pretrain.h5ad"
    elif args.basemodel_kind != "final":
        saturn_h5ad = (
            output_fdn / "saturn_results" / f"adata_ep_{args.basemodel_kind}.h5ad"
        )
    else:
        saturn_h5ad = output_fdn / "saturn_results" / "final_adata.h5ad"

    print("Read trained h5ad")
    adata_train = anndata.read_h5ad(saturn_h5ad)
    adata_train.obs["train_inf"] = "train"

    print("Read inference h5ad")
    termite_embeddings_summary_fdn = pathlib.Path(
        "/home/fabio/projects/compressed_atlas/cell_atlas_approximations_analysis/data/znev/esm1b_embeddings_summaries/"
    )
    termite_h5ad_fdn = pathlib.Path(
        "/home/fabio/projects/compressed_atlas/cell_atlas_approximations_analysis/data/znev/h5ads/"
    )
    inf_fdn = termite_embeddings_summary_fdn.parent / f"inference_output_{args.species}"
    for fn in inf_fdn.iterdir():
        if str(fn).endswith(f"_epoch_{args.finetune_epochs}_finetuned.h5ad"):
            inf_h5ad = fn
            break
    else:
        raise IOError(
            f"No SATURN-xfer inference h5ad for species {args.species} found in {inf_fdn}"
        )
    species_guide = "_".join(inf_h5ad.stem.split("_")[2:4])
    species_inf = species_guide + "-inf"
    adata_inf = anndata.read_h5ad(inf_h5ad)
    adata_inf.obs["train_inf"] = "inference"

    print("Limit training data to the guide/closest species")
    adata_train = adata_train[
        adata_train.obs["species"] == adata_inf.uns["guide_species"]
    ]

    print("Concatenate training and inference anndata")
    adata = anndata.concat([adata_train, adata_inf])

    print("Add tissue information")
    adata.obs["organ"] = ""
    separate_h5ad_fdn = output_fdn.parent / "h5ads"
    for species, datum in df.iterrows():
        if species not in adata_train.obs["species"].cat.categories:
            continue
        print(species)
        h5ad_fn = pathlib.Path(datum["path"])
        if str(h5ad_fn).startswith("/srv/scratch"):
            h5ad_fn = pathlib.Path(
                f"/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/h5ads/{species}.h5ad"
            )
        if not h5ad_fn.exists():
            species_orig = species.split("-")[0]
            h5ad_fn = pathlib.Path(
                f"/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/h5ads/{species_orig}.h5ad"
            )
        if not h5ad_fn.exists():
            print(f"File {h5ad_fn} does not exist, skipping")
            continue
        adatas = anndata.read_h5ad(h5ad_fn)
        cell_ids_species = adata.obs_names[adata.obs["species"] == species]
        organ_species = adatas.obs.loc[cell_ids_species, "organ"]
        adata.obs.loc[cell_ids_species, "organ"] = organ_species
        del adatas
    adata.obs["organ"] = pd.Categorical(adata.obs["organ"])
    __import__("gc").collect()

    print("Now we can make obs unique")
    adata.obs_names_make_unique()

    print("PCA")
    sc.pp.pca(adata)

    print("KNN")
    sc.pp.neighbors(adata)

    print("UMAP")
    sc.tl.umap(adata)

    print("Standardise some cell type names")

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

    if False:
        print("Plot UMAP with cell types in guide and inferred species")
        sc.pl.umap(adata, color="species", title="Species", add_outline=True, size=20)
        fig = plt.gcf()
        fig.set_size_inches(9, 5)
        fig.tight_layout()
        # fig.savefig("../figures/combined_umap_saturn_atlasapprox_species.png", dpi=300)

        # Same but only for guide and inf species
        palette = {key: "grey" for key in adata.obs["species"].cat.categories}
        palette[species_inf] = "tomato"
        palette[species_guide] = "steelblue"
        sc.pl.umap(
            adata,
            color="species",
            title="Species",
            groups=[species_inf, species_guide],
            palette=palette,
            add_outline=True,
            size=20,
        )
        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        fig.tight_layout()

        sc.pl.umap(adata, color="organ", title="Organ", add_outline=True, size=20)
        fig2 = plt.gcf()
        fig2.set_size_inches(10, 5)
        fig2.tight_layout()
        # fig2.savefig("../figures/combined_umap_saturn_atlasapprox_organ.png", dpi=300)

        cell_types = np.sort(adata.obs["cell_type"].unique())
        colors = sns.color_palette("husl", n_colors=len(cell_types))
        palette = dict(zip(cell_types, colors))
        sc.pl.umap(
            adata,
            color="cell_type",
            title="Cell Type",
            add_outline=True,
            size=15,
            palette=dict(zip(cell_types, colors)),
        )
        fig3 = plt.gcf()
        fig3.set_size_inches(17, 9.8)
        fig3.axes[0].legend(
            ncol=5,
            fontsize=6,
            bbox_to_anchor=(1, 1),
            bbox_transform=fig3.axes[0].transAxes,
        )
        fig3.tight_layout()
        # fig3.savefig("../figures/combined_umap_saturn_atlasapprox_celltype.png", dpi=300)
        #

    if True:
        print("Plot UMAP with the two species harmonised")
        palette = {
            full_name_dict[adata_inf.uns["guide_species"]].replace(" ", "\n"): (
                0.9,
                0.9,
                0.9,
            ),
            termite_full_names_dict[args.species].replace(" ", "\n"): "tomato",
        }
        adata.obs["species_full"] = adata.obs["species"].map(
            {
                adata_inf.uns["guide_species"]: full_name_dict[
                    adata_inf.uns["guide_species"]
                ].replace(" ", "\n"),
                f"{args.species}-inf": termite_full_names_dict[args.species].replace(
                    " ", "\n"
                ),
            }
        )

        fig, ax = plt.subplots(figsize=(4.5, 3))
        sc.pl.umap(
            adata,
            color="species_full",
            palette=palette,
            add_outline=True,
            size=20,
            ax=ax,
            title="",
            frameon=False,
        )
        ax.get_children()[-2].set_title("Organism:")
        fig.tight_layout()
        if args.savefig:
            fig.savefig(
                f"../figures/umap_roach_drosophila_species.svg",
            )
            fig.savefig(
                f"../figures/umap_roach_drosophila_species.png",
                dpi=300,
            )

    if True:
        print("Plot with only a few cell types at a time")
        fig3, axs = plt.subplots(2, 3, figsize=(16.5, 11), sharex=True, sharey=True)
        axs = axs.ravel()
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
            [
                adata_inf.uns["guide_species"],
                adata_inf.uns["guide_species"],
                adata_inf.uns["guide_species"],
                adata_inf.uns["guide_species"],
                f"{args.species}-inf",
            ]
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
            if i < 4:
                groups = cell_types[i::4] + ["other_species"]
            else:
                groups = cell_types
            sc.pl.umap(
                adata,
                color="cell_type_tmp",
                title=f"Cell Type ({species})",
                add_outline=True,
                size=15,
                palette=palette,
                ax=ax,
                groups=groups,
                na_color=palette["other species"],
            )
            ax.legend(ncol=2, fontsize=6, loc="best")
        for ax in axs:
            ax.set(xlabel=None, ylabel=None, xticks=[], yticks=[])
        fig3.tight_layout()
        if args.savefig:
            fig3.savefig(
                f"../figures/umap_znev_drosophila.svg",
            )
            fig3.savefig(
                f"../figures/umap_znev_drosophila.png",
                dpi=300,
            )

    if False:
        cell_types = np.sort(adata.obs["cell_type"].unique())
        colors = sns.color_palette("husl", n_colors=len(cell_types))
        palette = dict(zip(cell_types, colors))
        for i, ct in enumerate(cell_types):
            print(i + 1, ct)
            sc.pl.umap(
                adata,
                color="cell_type",
                title="Cell Type",
                groups=[ct],
                add_outline=True,
                size=18,
                palette=palette,
            )
            fig4 = plt.gcf()
            fig4.set_size_inches(8, 5)
            fig4.axes[0].legend(
                ncol=1,
                fontsize=6,
                bbox_to_anchor=(1, 1),
                bbox_transform=fig4.axes[0].transAxes,
            )
            # fig4.savefig(f"../figures/single_cell_types/{ct}.png", dpi=300)
            plt.close(fig4)

    # fig3, axs = plt.subplots(3, 4, figsize=(12, 9))
    # axs = axs.ravel()
    # palette = {
    #    1: 'tomato',
    #    0: (0.9, 0.9, 0.9, 0.001),
    # }
    # for species, ax in zip(species_full_dict.keys(), axs):
    #    adata.obs['is_focal'] = pd.Categorical((adata.obs['species'] == species).astype(int))
    #    sc.pl.umap(adata, color="is_focal", title=species_full_dict[species], add_outline=True, size=20, ax=ax, legend_loc=None, palette=palette, groups=[1], na_color=palette[0])
    # fig3.tight_layout()
    # fig3.savefig("../figures/combined_umap_saturn_all_species_first_try.png", dpi=600)

    # palette = {
    #    "soldier": "darkgrey",
    #    "worker": "purple",
    #    "king": "steelblue",
    #    "queen": "deeppink",
    #    "roach": "seagreen",
    # }
    # sc.pl.umap(adata, color="caste", title="Caste", add_outline=True, size=20, palette=palette)
    # fig4 = plt.gcf()
    # fig4.set_size_inches(6.5, 5)
    # fig4.tight_layout()
    # fig4.savefig("../figures/combined_umap_saturn_all_species_first_try_caste.png", dpi=600)

    print("Get closest annotations by cell type")
    from collections import Counter

    cdis = torch.cdist(
        torch.tensor(adata_inf.X).to("cuda"),
        torch.tensor(adata_train.X).to("cuda"),
    )
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

    tmp = close_dic_frac.stack()
    annotations = pd.DataFrame(
        {
            "annotation": close_dic_frac.idxmax(axis=1),
            "fraction": close_dic_frac.max(axis=1),
        }
    ).sort_values("fraction", ascending=False)

    print("Find how many annotations are clear")
    fr_clear = (annotations["fraction"] > 0.5).mean()
    print(f"% of clear (>50%) annotations: {fr_clear:.0%}")
    clear_clusters = annotations.index[annotations["fraction"] > 0.5]
    fr_cell_clear = adata_inf.obs["ref_labels"].isin(clear_clusters).mean()
    print(f"% of cells clearly annotated: {fr_cell_clear:.0%}")

    print("Find markers for neurons and muscle, to prove the point")
    __import__("os").environ["ATLASAPPROX_HIDECREDITS"] = "yes"
    import atlasapprox

    genes_to_macrogenes_fn = output_fdn / "saturn_results" / "genes_to_macrogenes.pkl"
    # FIXME: these do not include the inferred species, but we barely know anything about it anyway so we'll use the guide species genes for interpretation
    genes_to_macrogenes = pd.read_pickle(genes_to_macrogenes_fn)
    cell_types_verify = ["muscle", "epithelial"]
    # NOTE: neuron is bad in terms of assignment, but one can still see some neuronal genes there.
    fracs_by_cluster = pd.DataFrame(
        {
            ct: (
                adata_inf[adata_inf.obs["ref_labels"] == ct].obsm["macrogenes"] > 0
            ).mean(axis=0)
            for ct in annotations.index
        }
    )
    # NOTE: use human for interretation of macrogenes
    species_bait = "h_sapiens"
    var_names_bait = np.array(
        [
            x[len(species_bait) + 1 :]
            for x in genes_to_macrogenes
            if x.startswith(species_bait)
        ]
    )
    genes_to_macrogenes_bait_matrix = np.vstack(
        [genes_to_macrogenes[f"{species_bait}_{g}"] for g in var_names_bait],
    )
    clear_bets = ["TTN", "MYL1", "MYH11", "SYN3", "TTL", "CHIT1", "APOE"]
    # For TTL evidence for neurons: https://www.pnas.org/doi/10.1073/pnas.0409626102
    # NOTE: high expression of CHIT1 in roach epithelial cells, which is chitinase and degrades chitin
    if species_bait == "m_musculus":
        clear_bets = [x.capitalize() for x in clear_bets] + ["Slc1a1"]
    api = atlasapprox.API()
    for cell_type in cell_types_verify:
        print(cell_type)
        cluster = (
            annotations.loc[annotations["annotation"] == cell_type]
            .sort_values("fraction", ascending=False)
            .index[0]
        )
        # Find markers
        fracs_focal = fracs_by_cluster[cluster]
        fracs_other_max = fracs_by_cluster.drop(cluster, axis=1).max(axis=1)
        deltafr = fracs_focal - fracs_other_max
        markers_mg_fr = deltafr.nlargest(10)
        markers_mg_fr = markers_mg_fr[markers_mg_fr > 0]
        markers_mg = markers_mg_fr.index

        # Find what human genes correspond to these macrogenes
        genes_to_marker_mg = var_names_bait[
            genes_to_macrogenes_bait_matrix[:, markers_mg].argmax(axis=0)
        ]
        bait_genes = list(sorted(set(genes_to_marker_mg)))
        print(
            f"  {species_bait} genes influential for the marker macrogenes in this cell type group: ",
            ", ".join(bait_genes),
        )
        for gene in clear_bets:
            if gene in bait_genes:
                res = api.highest_measurement(
                    organism=species_bait,
                    number=5,
                    feature=gene,
                ).reset_index()
                res_hit = res.loc[res["celltype"].str.contains(cell_type)]
                print(gene)
                if len(res_hit) > 0:
                    print(res_hit)
                else:
                    print(res)

    if False:
        print("Load and plot UMAP with original cell annotations")
        termite_h5ad_fdn = pathlib.Path(
            "/mnt/data/projects/termites/data/sc_termite_data/saturn_data/h5ad_by_species"
        )
        h5ad_fn = termite_h5ad_fdn / termite_dict[args.species][0]
        adata_orig = anndata.read_h5ad(h5ad_fn)

        fig, ax = plt.subplots(figsize=(4.5, 3))
        sc.pl.umap(
            adata_orig,
            color="cell_type",
            title="Clusters",
            ax=ax,
            add_outline=True,
            size=20,
        )
        fig.tight_layout()
        if args.savefig:
            fig.savefig(
                f"../figures/umap_roach_original.svg",
            )
            fig.savefig(
                f"../figures/umap_roach_original.png",
                dpi=300,
            )

    if True:
        print("Plot UMAP with one cluster annotated and the guide assigned annotation")
        cell_types_verify = [
            "fat cell",
            "muscle cell",
            "neuron",
            "glial cell",
            "female reproductive system",
            "crystal cell",
            "T1",
            "T2",
            "T3",
            "T4",
        ]
        fig, axs = plt.subplots(3, 3, figsize=(20, 21), sharex=True, sharey=True)
        for ax, cell_type in zip(axs.ravel(), cell_types_verify):
            groups = [
                "znev-inf_" + cell_type,
                "d_melanogaster_" + annotations.at[cell_type, "annotation"],
            ]
            palette = {
                key: (0.9, 0.9, 0.9, 0.001)
                for key in adata.obs["labels"].cat.categories
            }
            palette["other"] = (0.9, 0.9, 0.9, 0.001)
            palette[groups[0]] = tuple(list(to_rgba("goldenrod")[:3]) + [0.1])
            palette[groups[1]] = tuple(list(to_rgba("tomato")[:3]) + [0.1])

            sc.pl.umap(
                adata,
                color="labels",
                ax=ax,
                add_outline=True,
                size=35,
                groups=groups,
                palette=palette,
                na_color=palette["other"],
                legend_loc="lower right",
                frameon=False,
                zorder=4,
            )
            # Curate legend
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            # Remove NA
            legend_handles = legend_handles[:-1]
            legend_labels = legend_labels[:-1]
            # Increase opacity
            for lh in legend_handles:
                lh.set_alpha(1.0)
            ax.legend(
                legend_handles,
                legend_labels,
                loc="best",
                fontsize=12,
                frameon=False,
            )

            ax.set_title("")
        fig.tight_layout()
        if args.savefig:
            fig.savefig(
                f"../figures/umap_znev_drosophila_matching_annotations.svg",
            )
            fig.savefig(
                f"../figures/umap_znev_drosophila_matching_annotations.png",
                dpi=300,
            )
