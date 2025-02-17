import os
import sys
import pathlib
import numpy as np
import pandas as pd
from scipy.special import erf
import anndata
import scquill

try:
    import requests

    requests.get("http://127.0.0.1:5000/v1/data_sources")
    os.environ["ATLASAPPROX_BASEURL"] = "http://localhost:5000"
except requests.exceptions.ConnectionError:
    pass
finally:
    os.environ["ATLASAPPROX_HIDECREDITS"] = "yes"

import atlasapprox

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns

data_fdn = pathlib.Path("__file__").resolve().parent.parent / "data" / "atlas_data"

blacklist_species = [
    "c_intestinalis",  # not sure what happened, needs updating
]
plants = [
    "a_thaliana",
    "t_aestivum",
    "z_mays",
    "l_minuta",
    "o_sativa",
    "f_vesca",
]


def get_atlas_path_dict():
    """Get a dictionary of atlas paths."""
    res = {}
    for fn in data_fdn.glob("*.h5"):
        if fn.stem not in blacklist_species:
            res[fn.stem] = fn
    return res


if __name__ == "__main__":

    adata_dict = {}
    atlas_path_dict = get_atlas_path_dict()
    for species, fn in atlas_path_dict.items():
        print(species)
        app = scquill.Approximation.read_h5(fn)
        adata = app.to_anndata(
            groupby=["tissue", "celltype"],
            measurement_type="gene_expression",
        )
        adata.obs["organism"] = species
        adata.obs_names = adata.obs[["tissue", "celltype"]].apply("->".join, axis=1)
        adata_dict[species] = adata

    # Check average above certain thresholds
    threshold = 0.02  # cptt
    distros = {}
    for species, adata in adata_dict.items():
        n_celltypes_expressing = (adata.X >= threshold).sum(axis=0)
        distros[species] = {
            "n_expressing": n_celltypes_expressing,
            "n_celltypes": adata.shape[0],
        }
        adata.var["n_expressing"] = n_celltypes_expressing
        adata.var["frac_expressing"] = 1.0 * n_celltypes_expressing / adata.shape[0]

    # Look at homologs across species
    def plot_homologs(species_src, species_tgt):
        api = atlasapprox.API()
        adatah = adata_dict[species_src]
        adatam = adata_dict[species_tgt]
        genes_src = []
        bins = [
            [0.0, 0.1],
            [0.1, 0.2],
            [0.2, 0.3],
            [0.4, 0.5],
            [0.5, 0.6],
            [0.6, 0.7],
            [0.7, 0.8],
            [0.8, 0.9],
            [0.9, 1.01],
        ]
        for binl, binr in bins:
            cands = adatah.var.query(
                f"{binl} <= frac_expressing < {binr}"
            ).index.values.copy()
            np.random.shuffle(cands)
            cands = cands[:30]
            genes_src += cands.tolist()
        homologs = api.homologs(
            source_organism=species_src,
            features=genes_src,
            target_organism=species_tgt,
        )
        # Reduce to closest homolog
        homologs_nr = homologs.loc[homologs.groupby("queries")["distances"].idxmin()]

        from scipy.stats import pearsonr, spearmanr

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Comparison of homologs from {species_src} to {species_tgt}")
        ax = axs[0]
        x = adatah.var.loc[homologs_nr["queries"], "frac_expressing"]
        y = adatam.var.loc[homologs_nr["targets"], "frac_expressing"]
        r = pearsonr(x, y)
        rho = spearmanr(x, y)
        ax.scatter(x - 3e-3, y - 3e-3, color="k")
        ax.set_xlabel(f"Fraction of cell types expressing [{species_src}]")
        ax.set_ylabel(f"Fraction of cell types expressing [{species_tgt}]")
        ax.set_xlim(1e-2, 1 - 2e-3)
        ax.set_ylim(1e-2, 1 - 2e-3)
        ax.set_xscale("logit")
        ax.set_yscale("logit")
        ax.grid(True)
        fig.tight_layout()

        ax = axs[1]
        homologs_nr[f"frac_{species_src}"] = x.values
        homologs_nr[f"frac_{species_tgt}"] = y.values
        homologs_nr["frac_diff"] = (
            homologs_nr[f"frac_{species_src}"] - homologs_nr[f"frac_{species_tgt}"]
        )
        x = np.abs(homologs_nr["frac_diff"])
        y = homologs_nr["distances"]
        r2 = pearsonr(x, y)
        rho2 = spearmanr(x, y)
        ax.scatter(
            np.abs(homologs_nr["frac_diff"]), homologs_nr["distances"], color="k"
        )
        ax.grid(True)
        ax.set_ylabel("PROST distance")
        ax.set_xlabel("|$\\Delta frac$|")
        fig.tight_layout()

        print("Pearson correlation of homolog fractions:", r)
        print("Pearson correlation of |Δfrac| vs PROST distance:", r2)

    plot_homologs("h_sapiens", "m_musculus")

    plt.ion()
    plt.show()

    # Check if markers are conserved
    species1 = "h_sapiens"
    species2 = "m_musculus"
    adata1 = adata_dict[species1]
    adata2 = adata_dict[species2]
    shared_celltypes = list(
        set(adata1.obs["celltype"].unique()) & set(adata2.obs["celltype"].unique())
    )

    def get_markers(adata, celltype, number=None):
        tissues = adata.obs["tissue"].unique()
        ranks = {}
        for tissue in tissues:
            idx_tissue = adata.obs_names[adata.obs["tissue"] == tissue]
            adata_tissue = adata[idx_tissue]
            if celltype not in adata_tissue.obs["celltype"].unique():
                continue
            idx_ct = adata_tissue.obs_names[adata_tissue.obs["celltype"] == celltype][0]
            idx_ot = [x for x in adata_tissue.obs_names if x != idx_ct]

            vec = np.asarray(adata_tissue[idx_ct].layers["fraction"][0])
            mat = np.asarray(adata_tissue[idx_ot].layers["fraction"])

            # These are the details
            diff = (vec - mat).mean(axis=0)

            diff = pd.Series(diff, index=adata_tissue.var_names)
            diff = diff.sort_values(ascending=False).to_frame("diff")
            diff["rank"] = np.arange(diff.shape[0])
            ranks_tissue = diff.loc[adata.var_names, "rank"]
            ranks[tissue] = ranks_tissue

        ranks = pd.DataFrame(ranks)
        ranks = ranks.loc[np.sqrt(ranks).sum(axis=1).sort_values().index]
        ranks["total"] = np.arange(ranks.shape[0])

        if number is not None:
            ranks = ranks.nsmallest(number, "total")
        else:
            ranks = ranks.sort_values("total")

        return ranks

    api = atlasapprox.API()

    r_log2fc_dist = []
    for celltype in shared_celltypes:
        ranks1 = get_markers(adata1, celltype)
        ranks2 = get_markers(adata2, celltype)
        nmarkers = 80

        homologs1 = api.homologs(
            source_organism=species1,
            features=ranks1.index[:nmarkers],
            target_organism=species2,
        )
        # Reduce to closest homolog
        homologs1 = homologs1.loc[homologs1.groupby("queries")["distances"].idxmin()]
        homologs1.rename(
            columns={"queries": species1, "targets": species2}, inplace=True
        )
        homologs1["source"] = species1
        homologs2 = api.homologs(
            source_organism=species2,
            features=ranks2.index[:nmarkers],
            target_organism=species1,
        )
        # Reduce to closest homolog
        homologs2 = homologs2.loc[homologs2.groupby("queries")["distances"].idxmin()]
        homologs2.rename(
            columns={"queries": species2, "targets": species1}, inplace=True
        )
        homologs2["source"] = species2

        homologs = pd.concat([homologs1, homologs2]).drop_duplicates(
            subset=[species1, species2]
        )
        homologs[f"rank_{species1}"] = ranks1.loc[homologs[species1], "total"].values
        homologs[f"rank_{species2}"] = ranks2.loc[homologs[species2], "total"].values
        homologs["log2fc"] = np.log2(homologs[f"rank_{species2}"] + 1) - np.log2(
            homologs[f"rank_{species1}"] + 1
        )

        if True:
            shared_tissues = list(
                set(ranks1.columns) & set(ranks2.columns) - set(["total"])
            )
            if len(shared_tissues) >= 3:
                homologs_strict = homologs.loc[
                    (homologs[f"rank_{species1}"] < 30)
                    | (homologs[f"rank_{species2}"] < 30)
                ]
                from scipy.spatial import ConvexHull

                fig, axs = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle(celltype)
                ax = axs[0]
                colors = sns.color_palette("husl", n_colors=len(homologs_strict))
                for i, (_, row) in enumerate(homologs_strict.iterrows()):
                    x = 1 + ranks1.loc[row[species1], shared_tissues]
                    y = 1 + ranks2.loc[row[species2], shared_tissues]
                    ax.scatter(
                        x,
                        y,
                        color=colors[i],
                        marker="s" if i % 2 else "o",
                        label=row[species1] + " / " + row[species2],
                        zorder=6,
                    )
                    ch = ConvexHull(np.vstack([x, y]).T)
                    ax.add_patch(
                        plt.Polygon(
                            ch.points[ch.vertices % len(x)],
                            closed=True,
                            fill=True,
                            edgecolor=colors[i],
                            linewidth=0.5,
                            facecolor=list(colors[i])[:3] + [0.1],
                            zorder=5,
                        ),
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(True)
                ax.set_xlabel(f"Marker rank in {species1}")
                ax.set_ylabel(f"Marker rank in {species2}")
                ax = axs[1]
                for i, (_, row) in enumerate(homologs_strict.iterrows()):
                    onames = [f"{t}->{celltype}" for t in shared_tissues]
                    x = np.asarray(1 + adata1[onames, row[species1]].X).ravel()
                    y = np.asarray(1 + adata2[onames, row[species2]].X).ravel()
                    ax.scatter(
                        x,
                        y,
                        color=colors[i],
                        marker="s" if i % 2 else "o",
                        label=row[species1] + " / " + row[species2],
                        zorder=6,
                    )
                    if y.max() == y.min():
                        y[0] *= 0.99
                    if x.max() == x.min():
                        x[0] *= 1.01
                    ch = ConvexHull(np.vstack([x, y]).T)
                    ax.add_patch(
                        plt.Polygon(
                            ch.points[ch.vertices % len(x)],
                            closed=True,
                            fill=True,
                            edgecolor=colors[i],
                            linewidth=0.5,
                            facecolor=list(colors[i])[:3] + [0.1],
                            zorder=5,
                        ),
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(True)
                ax.set_xlabel(f"Avg expr in {species1}")
                ax.set_ylabel(f"Avg expr in {species2}")
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1, 1),
                    bbox_transform=ax.transAxes,
                    ncols=2,
                )
                fig.tight_layout()

        if False:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(celltype)
            ax = axs[0]
            ax.scatter(
                1 + homologs[f"rank_{species1}"],
                1 + homologs[f"rank_{species2}"],
                color="k",
            )
            ax.set_xlabel(f"Marker rank in {species1}")
            ax.set_ylabel(f"Marker rank in {species2}")
            ax.grid(True)
            ax.set_xlim(left=0.8)
            ax.set_ylim(bottom=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            for _, row in homologs.iterrows():
                ax.text(
                    1 + row[f"rank_{species1}"],
                    1 + row[f"rank_{species2}"],
                    row[species1] + "\n" + row[species2],
                    fontsize=8,
                    ha="center",
                    va="center",
                )

            ax = axs[1]
            ax.scatter(np.abs(homologs["log2fc"]), homologs["distances"], color="k")
            ax.set_xlabel("|$\\Delta log2[rank]$|")
            ax.set_ylabel("PROST distance")
            ax.grid(True)
            fig.tight_layout()

        from scipy.stats import pearsonr

        r = pearsonr(homologs[f"rank_{species1}"], homologs[f"rank_{species2}"])
        r2 = pearsonr(np.abs(homologs["log2fc"]), homologs["distances"])
        print(celltype)
        print("Pearson correlation of marker ranks:", r)
        print("Pearson correlation of |Δlog2[rank]| vs PROST distance:", r2)

        r_log2fc_dist.append({"celltype": celltype, "r": r2[0], "pvalue": r2[1]})

    r_log2fc_dist = pd.DataFrame(r_log2fc_dist).set_index("celltype")
    r_log2fc_dist["neglog10_pvalue"] = -np.log10(r_log2fc_dist["pvalue"])
    r_log2fc_dist = r_log2fc_dist.sort_values("neglog10_pvalue", ascending=False)

    fig, ax = plt.subplots(figsize=(5, 8))
    ax.barh(
        np.arange(r_log2fc_dist.shape[0]), r_log2fc_dist["neglog10_pvalue"], color="k"
    )
    ax.set_yticks(np.arange(r_log2fc_dist.shape[0]))
    ax.set_yticklabels(r_log2fc_dist.index)
    ax.set_xlabel("-log10(p-value)")
    ax.set_ylim(r_log2fc_dist.shape[0], -1)
    ax.axvline(3, ls="--", color="tomato")
    ax.set_title("Significance of correlation between\nmarker rank and PROST distance")
    fig.tight_layout()
