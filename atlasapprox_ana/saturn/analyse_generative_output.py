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
import matplotlib

matplotlib.use("Qt5Agg")
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
roach_dict = {
    # Imperfect genome matches
    "cpun": ["cpun.h5ad", "Cmer_gene_all_esm1b.pt"],
}
roach_full_names_dict = {
    "cpun": "Cryptocercus punctulatus",
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyse the output of SATURN.")
    parser.add_argument(
        "--species",
        type=str,
        default="t_cynocephalus",
        help="Species that was inferred",
    )
    parser.add_argument(
        "--kind",
        choices=["zeroshot", "finetuned"],
        default="zeroshot",
        type=str,
        help="Whether to look at zeroshot of finetuned trained model output",
    )
    parser.add_argument(
        "--savefig",
        action="store_true",
        help="Whether to save the figures",
    )
    args = parser.parse_args()

    h5ad_fdn = pathlib.Path(
        f"/mnt/data/projects/cell_atlas_approximations/noatlas_species/synthetic_atlases/"
    )
    for fn in h5ad_fdn.iterdir():
        fnstr = str(fn)
        if (args.species in fnstr) and (args.kind in fnstr) and fnstr.endswith(".h5ad"):
            h5ad_fn = fn
            break
    else:
        sys.exit(f"h5ad file for species {args.species} not found in {h5ad_fdn}")

    print("Look for processed h5ad if available")
    h5ad_proc_fn = h5ad_fn.with_name(h5ad_fn.stem + "_processed.h5ad")
    if h5ad_proc_fn.exists():
        print("Read generated atlas (processed)")
        adata_gen = anndata.read_h5ad(h5ad_proc_fn)
    else:
        print("Read generated atlas (raw)")
        adata_gen = anndata.read_h5ad(h5ad_fn)

        print("Compute HVGs")
        sc.pp.highly_variable_genes(adata_gen, flavor="seurat_v3")

        print("PCA")
        sc.pp.pca(adata_gen)

        print("KNN")
        sc.pp.neighbors(adata_gen)

        print("UMAP")
        sc.tl.umap(adata_gen)

        print("Marker genes")
        sc.tl.rank_genes_groups(
            adata_gen,
            groupby="ref_labels",
            groups=[
                "smooth muscle",
                "striated muscle",
                "macrophage",
                "neuron",
                "B",
                "T",
                "NK",
                "capillary",
                "epithelial",
            ],
            method="wilcoxon",
        )

        print("Store processed h5ad to disk")
        adata_gen.write(h5ad_proc_fn)

    print("Load protein embeddings for homology search")
    if args.species == "t_cynocephalus":
        prot_emb_fn = f"/mnt/data/projects/cell_atlas_approximations/noatlas_species/esm1b_embeddings_summaries/{args.species}_gene_all_esm1b.pt"
    else:
        prot_emb_fn = f"/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/esm_embeddings_summaries/{args.species}_gene_all_esm1b.pt"
    prot_emb_dict = torch.load(prot_emb_fn)
    species_genes = adata_gen.var_names
    prot_emb = torch.stack([prot_emb_dict[g] for g in species_genes])
    prot_emb_human_fn = "/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/esm_embeddings_summaries/h_sapiens_gene_all_esm1b.pt"
    prot_emb_human_dict = torch.load(prot_emb_human_fn)
    human_genes = pd.Index(list(prot_emb_human_dict.keys()))
    prot_emb_human = torch.stack([prot_emb_human_dict[g] for g in human_genes])
    gene_ser_dict = {
        args.species: pd.Series(np.arange(len(species_genes)), index=species_genes),
    }
    if args.species != "h_sapiens":
        gene_ser_dict["h_sapiens"] = (
            pd.Series(np.arange(len(human_genes)), index=human_genes),
        )
    cdis_to_human = torch.cdist(prot_emb.to("cuda"), prot_emb_human.to("cuda"))

    plt.ion()

    if False:
        print("Plot UMAP by cell type, all together")
        fig, ax = plt.subplots(figsize=(15, 6))
        sc.pl.umap(
            adata_gen,
            color="ref_labels",
            ax=ax,
            size=50,
            title=f"{args.species} cell types",
        )
        ax.set_axis_off()
        fig.tight_layout()
        if args.savefig:
            fig.savefig(
                f"../../figures/{args.species}/{args.species}_umap_all_cell_types.png",
                dpi=300,
            )

    if args.species != "t_cynocephalus":
        print("Plot UMAP by cell type, in chunks")
        plt.ion()
        lst = adata_gen.obs["ref_labels"].cat.categories
        n = len(lst) // 15
        chunks = [lst[i : i + n] for i in range(0, len(lst), n)]
        fig, axs = plt.subplots(3, 5, figsize=(25, 15), sharex=True, sharey=True)
        for ax, ctsi in zip(axs.ravel(), chunks):
            sc.pl.umap(
                adata_gen,
                color="ref_labels",
                groups=list(ctsi),
                na_color=(0.9, 0.9, 0.9, 0.001),
                ax=ax,
                size=50,
            )
            ax.set_axis_off()
        fig.tight_layout()

    print("Load cell supertypes for the guide species")
    import yaml

    if args.species == "t_cynocephalus":
        guide_species = "m_musculus"
    else:
        guide_species = args.species
    with open(
        f"../../../cell_atlas_approximations_compression/compression/organism_configs/{guide_species}.yml"
    ) as f:
        tmp = yaml.safe_load(f)
    cell_supertypes = tmp["cell_annotations"]["cell_supertypes"]
    supertype_dict = {v: k for k, values in cell_supertypes.items() for v in values}
    adata_gen.obs["cell_supertype"] = pd.Categorical(
        adata_gen.obs["ref_labels"].map(supertype_dict)
    )

    print("Plot UMAP by cell supertype")
    if args.species != "t_cynocephalus":
        width, height = 4.3, 3
    else:
        width, height = 4.5, 4.5
    fig, ax = plt.subplots(figsize=(width, height))
    palette = {
        "endothelial": "tomato",
        "epithelial": "violet",
        "mesenchymal": "steelblue",
        "immune": "yellowgreen",
        "other": "gold",
    }
    kwargs = {}
    if args.species == "t_cynocephalus":
        kwargs["legend_loc"] = "lower right"
    sc.pl.umap(
        adata_gen,
        color="cell_supertype",
        ax=ax,
        size=50,
        add_outline=True,
        palette=palette,
        alpha=0.2,
        title="",
        **kwargs,
    )
    ax.set_axis_off()
    fig.tight_layout()
    if args.savefig:
        fig.savefig(
            f"../../figures/{args.species}/{args.species}_umap_cell_supertypes.png",
            dpi=300,
        )

    if False:
        print("Plot distribution of distances between and within supertypes")
        supertypes = adata_gen.obs["cell_supertype"].cat.categories
        nst = len(supertypes)
        fig, axs = plt.subplots(nst, nst, figsize=(5, 5), sharex=True, sharey=True)
        for i1, (st1, axrow) in enumerate(zip(supertypes, axs)):
            idx1 = adata_gen.obs["cell_supertype"] == st1
            axrow[0].set_ylabel(st1)
            for i2, (st2, ax) in enumerate(zip(supertypes, axrow)):
                idx2 = adata_gen.obs["cell_supertype"] == st2
                cdis = (
                    torch.cdist(
                        torch.tensor(adata_gen.X[idx1]).to("cuda"),
                        torch.tensor(adata_gen.X[idx2]).to("cuda"),
                    )
                    .to("cpu")
                    .numpy()
                    .ravel()
                )
                ax.ecdf(cdis, complementary=True)
                if i1 == nst - 1:
                    ax.set_xlabel(st2)
        fig.tight_layout()

    if False:
        print(
            "Plot distribution of distances between and within supertypes, simplified"
        )
        idx1, idx2 = np.random.choice(adata_gen.n_obs, 2000, replace=False).reshape(
            (2, 1000)
        )
        dis = torch.tensor(adata_gen.X[idx1]).to("cuda") - torch.tensor(
            adata_gen.X[idx2]
        ).to("cuda")
        dis = (dis * dis).sum(axis=1).sqrt().to("cpu").numpy()
        df = pd.DataFrame({"distance": dis})
        df["supertype1"] = adata_gen.obs["cell_supertype"].values[idx1]
        df["supertype2"] = adata_gen.obs["cell_supertype"].values[idx2]
        df["same_supertype"] = df["supertype1"] == df["supertype2"]

        fig, ax = plt.subplots(figsize=(5, 2))
        x = np.sort(df.loc[df["same_supertype"], "distance"].values)
        y = 1.0 - np.linspace(0, 1, len(x))
        ax.plot(x, y, label="same", color="tomato")
        x = np.sort(df.loc[~df["same_supertype"], "distance"].values)
        y = 1.0 - np.linspace(0, 1, len(x))
        ax.plot(x, y, label="between", color="black")
        ax.legend()
        fig.tight_layout()

    print("Plot distribution of nearest neighbors")
    idx = np.random.choice(adata_gen.n_obs, 1000, replace=False)
    neis = adata_gen.obsp["connectivities"][idx]
    idxn = np.asarray(neis.argmax(axis=1)).ravel()
    st1 = adata_gen.obs["cell_supertype"].values[idx]
    stn = adata_gen.obs["cell_supertype"].values[idxn]
    df = pd.DataFrame({"st1": st1, "stn": stn})
    df["c"] = 1
    gby = df.groupby(["st1", "stn"], observed=True).size().unstack(fill_value=0)
    frac = {g: gby.loc[g, g] / gby.loc[g].sum() for g in gby.index}
    frac_null = []
    for rep in range(100):
        idx1, idxn = np.random.choice(adata_gen.n_obs, 2000, replace=False).reshape(
            (2, 1000)
        )
        st1 = adata_gen.obs["cell_supertype"].values[idx]
        stn = adata_gen.obs["cell_supertype"].values[idxn]
        df = pd.DataFrame({"st1": st1, "stn": stn})
        df["c"] = 1
        gby = df.groupby(["st1", "stn"], observed=True).size().unstack(fill_value=0)
        frac_null.append({g: gby.loc[g, g] / gby.loc[g].sum() for g in gby.index})
    frac_null = pd.DataFrame(frac_null)

    from scipy.stats import gaussian_kde

    palette = {
        "endothelial": "tomato",
        "epithelial": "violet",
        "mesenchymal": "steelblue",
        "immune": "yellowgreen",
        "other": "gold",
    }
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    xticklabels = []
    yarr = np.linspace(0, 1, 200)
    for i, (st, fr) in enumerate(frac.items()):
        ax.scatter(
            [i], [100 * fr], s=70, facecolor=palette[st], edgecolor="darkgrey", lw=2
        )
        harr = gaussian_kde(frac_null[st].values)(yarr)
        harr /= 2.5 * harr.max()
        idxtmp = harr > 0.01
        ax.fill_betweenx(
            100 * yarr[idxtmp],
            i - harr[idxtmp],
            i + harr[idxtmp],
            color=palette[st],
            alpha=0.3,
        )
        xticklabels.append(st)
    ax.set_xticks(range(len(frac)))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("% neighbors\nwithin supertype")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if args.savefig:
        fig.savefig(
            f"../../figures/{args.species}/{args.species}_supertype_neighbors_stats.svg",
        )
        fig.savefig(
            f"../../figures/{args.species}/{args.species}_supertype_neighbors_stats.png",
            dpi=300,
        )

    print("Plot distribution of nearest neighbors in individual cell types")
    idx = np.random.choice(adata_gen.n_obs, 4000, replace=False)
    neis = adata_gen.obsp["connectivities"][idx]
    idxn = np.asarray(neis.argmax(axis=1)).ravel()
    st1 = adata_gen.obs["ref_labels"].values[idx]
    stn = adata_gen.obs["ref_labels"].values[idxn]
    df = pd.DataFrame({"st1": st1, "stn": stn})
    df["c"] = 1
    gby = df.groupby(["st1", "stn"], observed=True).size().unstack(fill_value=0)
    frac = {g: gby.loc[g, g] / gby.loc[g].sum() for g in gby.index if g in gby.columns}
    frac_null = []
    for rep in range(100):
        idx1, idxn = np.random.choice(adata_gen.n_obs, 8000, replace=False).reshape(
            (2, 4000)
        )
        st1 = adata_gen.obs["ref_labels"].values[idx]
        stn = adata_gen.obs["ref_labels"].values[idxn]
        df = pd.DataFrame({"st1": st1, "stn": stn})
        df["c"] = 1
        gby = df.groupby(["st1", "stn"], observed=True).size().unstack(fill_value=0)
        frac_null.append(
            {g: gby.loc[g, g] / gby.loc[g].sum() for g in gby.index if g in gby.columns}
        )
    frac_null = pd.DataFrame(frac_null)

    from scipy.stats import gaussian_kde

    palette = {
        "monocyte": "tomato",
        "macrophage": "grey",
        "T": "gold",
        "B": "violet",
        "mast": "yellowgreen",
        "striated muscle": "steelblue",
        "smooth muscle": "navy",
        "vascular smooth muscle": "cadetblue",
        "pericyte": "deeppink",
        "brush": "tan",
        "basal": "peru",
        "ciliated": "chocolate",
        "HSC": "black",
        "erythrocyte": "red",
        "neuron": "blue",
        "hepatocyte": "mediumpurple",
        "alpha": "maroon",
        "beta": "lightcoral",
    }
    fig, ax = plt.subplots(figsize=(1 + 0.4 * len(palette), 2.5))
    xticklabels = []
    yarr = np.linspace(0, 1, 200)
    for i, (st, color) in enumerate(palette.items()):
        fr = frac[st]
        ax.scatter(
            [i], [100 * fr], s=70, facecolor=palette[st], edgecolor="darkgrey", lw=2
        )
        frtmp = frac_null[st].values
        frtmp = frtmp[~np.isnan(frtmp)]
        if len(np.unique(frtmp)) > 1:
            harr = gaussian_kde(frtmp)(yarr)
        else:
            harr = 0 * yarr
            harr[(yarr - frac_null[st].values[0]) ** 2 < 0.01] = 1
        harr /= 2.5 * harr.max()
        idxtmp = harr > 0.01
        ax.fill_betweenx(
            100 * yarr[idxtmp],
            i - harr[idxtmp],
            i + harr[idxtmp],
            color=palette[st],
            alpha=0.3,
        )
        xticklabels.append(st)
    ax.set_xticks(range(len(palette)))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 50, 100])
    ax.grid(True, axis="y")
    ax.set_ylabel("% neighbors\nwithin type")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if args.savefig:
        fig.savefig(
            f"../../figures/{args.species}/{args.species}_celltype_neighbors_stats.svg",
        )
        fig.savefig(
            f"../../figures/{args.species}/{args.species}_celltype_neighbors_stats.png",
            dpi=300,
        )

    if args.species != "t_cynocephalus":
        print(
            "Plot distribution of nearest neighbors in individual cell types compared to original atlas"
        )
        import glob

        orig_atlas_fns = glob.glob(
            "/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/curated_atlases/h_sapiens_*gene_expression.h5ad"
        )
        print("  Read original cell atlas")
        adata_origd = {}
        for fn in orig_atlas_fns:
            tissue = pathlib.Path(fn).stem.split("_")[2]
            adata_origd[tissue] = anndata.read_h5ad(fn)
            adata_origd[tissue].obs["tissue"] = tissue
        if len(adata_origd) > 1:
            print("  Concatenate tissues")
            adata_orig = anndata.concat(adata_origd.values())
        else:
            adata_orig = list(adata_origd.values())[0]

        print("  Preprocess original atlas (norm, PCA, etc)")
        sc.pp.highly_variable_genes(adata_orig, flavor="cell_ranger")
        sc.pp.normalize_total(adata_orig, target_sum=1e4)
        sc.pp.pca(adata_orig)
        sc.pp.neighbors(adata_orig)

        print("  Finally, compute the neighbors fraction in original atlas")
        idx = np.random.choice(adata_orig.n_obs, 6000, replace=False)
        neis = adata_orig.obsp["connectivities"][idx]
        idxn = np.asarray(neis.argmax(axis=1)).ravel()
        st1 = adata_orig.obs["cellType"].values[idx]
        stn = adata_orig.obs["cellType"].values[idxn]
        df = pd.DataFrame({"st1": st1, "stn": stn})
        df["c"] = 1
        gby_orig = (
            df.groupby(["st1", "stn"], observed=True).size().unstack(fill_value=0)
        )
        frac_orig = {
            g: gby_orig.loc[g, g] / gby_orig.loc[g].sum()
            for g in gby_orig.index
            if g in gby_orig.columns
        }

        palette = {
            "monocyte": "tomato",
            "macrophage": "grey",
            "T": "gold",
            "B": "violet",
            "mast": "yellowgreen",
            "striated muscle": "steelblue",
            "smooth muscle": "navy",
            "vascular smooth muscle": "cadetblue",
            "pericyte": "deeppink",
            "brush": "tan",
            "basal": "peru",
            "ciliated": "chocolate",
            "HSC": "black",
            "erythrocyte": "red",
            "neuron": "blue",
            "hepatocyte": "mediumpurple",
            "alpha": "maroon",
            "beta": "lightcoral",
            "acinar": "chartreuse",
            "capillary": "darkgreen",
            "CAP2": "darkolivegreen",
            "arterial": "salmon",
            "venous": "darkslateblue",
        }
        palette = {
            key: value
            for key, value in palette.items()
            if key in frac and key in frac_orig
        }
        tmp = pd.DataFrame({"orig": frac_orig, "synt": frac}).loc[list(palette.keys())]
        ct_order = (tmp["orig"] - tmp["synt"]).sort_values().index
        fig, ax = plt.subplots(figsize=(1 + 0.4 * len(palette), 3.5))
        xticklabels = []
        yarr = np.linspace(0, 1, 200)
        for i, st in enumerate(ct_order):
            color = palette[st]
            fr = frac[st]
            fr_orig = frac_orig[st]
            ax.scatter(
                [i],
                [100 * fr],
                s=70,
                facecolor=palette[st],
                edgecolor="darkgrey",
                lw=2,
                clip_on=False,
                zorder=10,
            )
            ax.scatter(
                [i],
                [100 * fr_orig],
                s=70,
                marker="s",
                facecolor=palette[st],
                edgecolor="darkgrey",
                lw=2,
                clip_on=False,
                zorder=9,
            )
            if abs(fr - fr_orig) > 0.20:
                sgn = (fr > fr_orig) * 2 - 1
                y0 = fr_orig + sgn * 0.08
                y1 = fr - sgn * 0.1
                ax.arrow(
                    i,
                    100 * y0,
                    0,
                    100 * (y1 - y0),
                    head_width=0.2,
                    head_length=5,
                    fc=color,
                )

            frtmp = frac_null[st].values
            frtmp = frtmp[~np.isnan(frtmp)]
            if len(np.unique(frtmp)) > 1:
                harr = gaussian_kde(frtmp)(yarr)
            else:
                harr = 0 * yarr
                harr[(yarr - frac_null[st].values[0]) ** 2 < 0.01] = 1
            harr /= 2.5 * harr.max()
            idxtmp = harr > 0.01
            ax.fill_betweenx(
                100 * yarr[idxtmp],
                i - harr[idxtmp],
                i + harr[idxtmp],
                color=palette[st],
                alpha=0.3,
            )
            xticklabels.append(st)
        ax.set_xticks(range(len(palette)))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 50, 100])
        ax.grid(True, axis="y")
        ax.set_ylabel("% cell neighbors\nwithin cell type")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        if args.savefig:
            fig.savefig(
                f"../../figures/{args.species}/{args.species}_celltype_neighbors_stats_with_orig.svg",
            )
            fig.savefig(
                f"../../figures/{args.species}/{args.species}_celltype_neighbors_stats_with_orig.png",
                dpi=300,
            )

    if False:
        print("Plot UMAP with marker genes")
        # Macrophages: one can use SLC12A1 as an example that it's not totally random, but hard
        for ct in ["B", "macrophage", "smooth muscle", "striated muscle", "neuron"]:
            fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
            markers = adata_gen.uns["rank_genes_groups"]["names"][ct][
                : len(axs.ravel()) - 1
            ]
            axs = axs.ravel()
            sc.pl.umap(
                adata_gen,
                color="ref_labels",
                groups=[ct],
                na_color=(0.9, 0.9, 0.9, 0.01),
                ax=axs[0],
                size=50,
            )
            axs[0].set_axis_off()
            for ax, gene in zip(axs[1:], markers):
                species_idx = gene_ser_dict[args.species][gene]
                human_idx = (
                    cdis_to_human[species_idx]
                    .topk(3, largest=False)
                    .indices.cpu()
                    .numpy()
                )
                human_homologs = ",".join(human_genes[human_idx])
                sc.pl.umap(
                    adata_gen,
                    color=gene,
                    ax=ax,
                    size=50,
                    title=f"{gene}\n({human_homologs})",
                )
                ax.set_axis_off()
            fig.suptitle(ct)
            fig.tight_layout()
            break
            if args.savefig:
                fig.savefig(
                    f"../../figures/{args.species}/{args.species}_umap_{ct}_markers_with_human_homologs.png",
                    dpi=300,
                )

    if args.species == "t_cynocephalus":
        print("Plot UMAP with select marker genes")
        shortlist = [
            ("striated muscle", "rna-TRANSCRIPT_031960541.1"),
            # ("macrophage", "rna-TRANSCRIPT_012547249.2"),
            ("B", "rna-TRANSCRIPT_031962951.1"),
        ]
        for ct, gene in shortlist:
            palette = {
                key: "grey" for key in adata_gen.obs["ref_labels"].cat.categories
            }
            palette[ct] = "tomato"
            fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))
            sc.pl.umap(
                adata_gen,
                color="ref_labels",
                groups=[ct],
                palette=palette,
                na_color=(0.9, 0.9, 0.9, 0.01),
                ax=axs[0],
                size=50,
                title=ct,
                na_in_legend=False,
                legend_loc=None,
            )
            axs[0].set_axis_off()
            ax = axs[1]
            species_idx = gene_ser_dict[args.species][gene]
            human_idx = (
                cdis_to_human[species_idx].topk(3, largest=False).indices.cpu().numpy()
            )
            human_homologs = ",".join(human_genes[human_idx])
            sc.pl.umap(
                adata_gen,
                color=gene,
                ax=ax,
                size=50,
                title=f"{gene}\n(human {human_homologs})",
            )
            ax.set_axis_off()
            fig.tight_layout()
            if args.savefig:
                fig.savefig(
                    f"../../figures/{args.species}/{args.species}_umap_{gene}_with_human_homologs.png",
                    dpi=300,
                )

    if True:
        print("Plot UMAP with human marker genes")
        human_marker_dict = {
            "B": ["MS4A1", "CD79A", "CD79B", "CD19"],
            # "striated muscle": ["ENO3", "MYL1", "SLN", "MYBPC1"],
            # "macrophage": ["MARCO", "CXCL3", "MRC1", "CD68"],
        }
        for ct, hmarkers in human_marker_dict.items():
            fig, axs = plt.subplots(4, 3, figsize=(18, 14), sharex=True, sharey=True)
            for axrow, gene in zip(axs, hmarkers):
                human_idx = gene_ser_dict["h_sapiens"][gene]
                species_idx = (
                    cdis_to_human.t()[human_idx]
                    .topk(3, largest=False)
                    .indices.cpu()
                    .numpy()
                )
                species_homologs = species_genes[species_idx]
                for i, (ax, homolog) in enumerate(zip(axrow, species_homologs)):
                    d = cdis_to_human[species_idx[i], human_idx].cpu().numpy()
                    sc.pl.umap(
                        adata_gen,
                        color=homolog,
                        ax=ax,
                        size=50,
                        title=f"{homolog},d={d:.2f}",
                    )
                    ax.set_axis_off()
                ax = axrow[0]
                ax.set_axis_on()
                ax.xaxis.set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.set_ylabel(gene)
            fig.suptitle(ct)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            if args.savefig:
                fig.savefig(
                    f"../../figures/{args.species}/{args.species}_umap_{ct}_reverse_markers_from_human.png",
                    dpi=300,
                )
            break

    sys.exit()

    saturn_csv_fn = output_fdn / "in_csv.csv"
    df = pd.read_csv(saturn_csv_fn, index_col="species")

    if args.kind == "pretrain":
        saturn_h5ad = output_fdn / "saturn_results" / "adata_pretrain.h5ad"
    elif args.kind != "final":
        saturn_h5ad = output_fdn / "saturn_results" / f"adata_ep_{args.kind}.h5ad"
    else:
        saturn_h5ad = output_fdn / "saturn_results" / "final_adata.h5ad"

    print("Read trained h5ad")
    adata_train = anndata.read_h5ad(saturn_h5ad)
    adata_train.obs["train_inf"] = "train"

    print("Read inference h5ad")
    inf_fdn = output_fdn / f"inference_output_{args.species}"
    for fn in inf_fdn.iterdir():
        if str(fn).endswith(f"_epoch_{args.finetune_epochs}_finetuned.h5ad"):
            inf_h5ad = fn
            break
    else:
        raise IOError(f"No inference h5ad found in {inf_fdn}")
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
            if species.endswith("-inf"):
                cell_types = sorted(cell_types, key=int)
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
                f"../figures/umap_roach_drosophila.svg",
            )
            fig3.savefig(
                f"../figures/umap_roach_drosophila.png",
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

    if True:
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
        cell_types_verify = ["muscle", "neuron"]
        fig, axs = plt.subplots(1, 2, figsize=(7, 3.5), sharex=True, sharey=True)
        for ax, cell_type in zip(axs, cell_types_verify):
            groups = [
                cell_type,
            ] + list(
                annotations.index[annotations["annotation"] == cell_type]
            )[:1]
            palette = {
                key: (0.9, 0.9, 0.9, 0.001)
                for key in adata.obs["ref_labels"].cat.categories
            }
            palette[cell_type] = "goldenrod"
            palette[groups[1]] = "tomato"

            sc.pl.umap(
                adata[~adata.obs["ref_labels"].isin(groups)],
                color="ref_labels",
                ax=ax,
                add_outline=True,
                size=20,
                groups=[],
                palette=palette,
                na_color=palette["0"],
                legend_loc=None,
                frameon=False,
                zorder=3,
            )

            sc.pl.umap(
                adata[adata.obs["ref_labels"].isin(groups)],
                color="ref_labels",
                ax=ax,
                add_outline=True,
                size=25,
                groups=groups,
                palette=palette,
                na_color=palette["0"],
                legend_loc="lower right",
                frameon=False,
                zorder=4,
            )
            ax.set_title("")
        fig.tight_layout()
        if args.savefig:
            fig.savefig(
                f"../figures/umap_roach_drosophila_matching_annotations.svg",
            )
            fig.savefig(
                f"../figures/umap_roach_drosophila_matching_annotations.png",
                dpi=300,
            )
