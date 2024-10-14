import os
import sys
import pathlib
import numpy as np
import pandas as pd
from scipy.special import erf
import anndata
import scquill

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
        adata.obs_names = adata.obs[["organism", "tissue", "celltype"]].apply(
            "->".join, axis=1
        )
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

    order = sorted(
        distros.keys(),
        key=lambda x: (distros[x]["n_expressing"] > 0.8).mean(),
        reverse=True,
    )
    colord = sns.color_palette("husl", n_colors=len(distros))
    colord = dict(zip(order, colord))

    # This is tricky because of pseudogenes that are technically present but never expressed
    if False:
        fig, axs = plt.subplots(2, 1, figsize=(8, 9), sharex=True, sharey=True)
        for species in order:
            distro = distros[species]
            x = 1.0 * distro["n_expressing"] / distro["n_celltypes"]
            ax = axs[int(distro["n_celltypes"] > 100)]
            ax.ecdf(x, complementary=True, label=species, color=colord[species])
        for ax in axs:
            ax.legend(
                ncol=2,
                loc="upper left",
                bbox_to_anchor=(1, 1),
                bbox_transform=ax.transAxes,
                title="Species",
            )
            ax.grid(True)
            ax.set_ylabel("Fraction of genes")
        axs[-1].set_xlabel("Fraction of cell types expressing gene")
        fig.tight_layout()

    # Repeat the above but with only genes that are acutally expressed by >= 20% cell types
    fig, axs = plt.subplots(3, 1, figsize=(8, 14), sharex=True, sharey=True)
    for species in order:
        distro = distros[species]
        x = 1.0 * distro["n_expressing"] / distro["n_celltypes"]
        x = x[x >= 0.2]
        if species in plants:
            ax = axs[0]
        else:
            ax = axs[1 + int(distro["n_celltypes"] > 100)]
        ax.ecdf(x, complementary=True, label=species, color=colord[species])
    for ax in axs:
        ax.legend(
            ncol=2,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            bbox_transform=ax.transAxes,
            title="Species",
        )
        ax.grid(True)
        ax.set_ylabel("Fraction of genes")
        x = [0.2, 1.0]
        y = [1.0, 0.0]
        ax.plot(x, y, color="k", linestyle="--")
    axs[-1].set_xlabel("Fraction of cell types expressing gene")
    fig.tight_layout()

    # Look at number o genes rather than fraction
    fig, axs = plt.subplots(3, 1, figsize=(8, 14), sharex=True, sharey=True)
    for species in order:
        distro = distros[species]
        x = 1.0 * distro["n_expressing"] / distro["n_celltypes"]
        x = np.sort(x)
        y = (1.0 - np.linspace(0, 1, len(x))) * len(x)
        if species in plants:
            ax = axs[0]
        else:
            ax = axs[1 + int(distro["n_celltypes"] > 100)]
        ax.plot(x, y, label=species, color=colord[species])
    for ax in axs:
        ax.legend(
            ncol=2,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            bbox_transform=ax.transAxes,
            title="Species",
        )
        ax.grid(True)
        ax.set_ylabel("Number of genes")
        ax.set_ylim(0, 30000)
    axs[-1].set_xlabel("Fraction of cell types expressing gene")
    fig.tight_layout()

    # Compare different tissues in a species
    from collections import defaultdict

    distros_tissues = defaultdict(dict)
    for species, adata in adata_dict.items():
        if adata.obs["tissue"].nunique() < 2:
            continue
        for tissue in adata.obs["tissue"].unique():
            adatat = adata[adata.obs["tissue"] == tissue]
            n_celltypes_expressing = (adatat.X >= threshold).sum(axis=0)
            distros_tissues[species][tissue] = {
                "n_expressing": n_celltypes_expressing,
                "n_celltypes": adatat.shape[0],
            }

    def fit_alpha_sigmoidal(distrosi, tissues=None):
        from scipy.optimize import minimize

        # Data from all tissues
        xs = []
        ys = []
        for tissue in order:
            if tissues is not None and tissue not in tissues:
                continue
            distro = distrosi[tissue]
            x = 1.0 * distro["n_expressing"] / distro["n_celltypes"]
            x = np.sort(x)
            y = np.linspace(0, 1, len(x))
            xs.append(x)
            ys.append(y)
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)

        # Function to minimise
        def fun_min(p):
            alpha, ymid = p
            xfit = 0.5 * (1.0 + erf(alpha * (ys - ymid) / np.sqrt(2)))
            res = ((xfit - xs) ** 2).sum()
            return res

        fit = minimize(fun_min, [1.0, 0.5], bounds=[(0.2, 30), (0.1, 0.9)])

        return fit.x

    alphas_all = []
    fig, axs = plt.subplots(4, 2, figsize=(12, 16), sharex=True)
    axs = axs.ravel()
    for species, ax in zip(distros_tissues.keys(), axs):
        distrosi = distros_tissues[species]
        order = sorted(
            distrosi.keys(),
            key=lambda x: (distrosi[x]["n_expressing"] > 0.8).mean(),
            reverse=True,
        )
        colord = sns.color_palette("husl", n_colors=len(distrosi))
        colord = dict(zip(order, colord))

        for tissue in order:
            distro = distrosi[tissue]
            x = 1.0 * distro["n_expressing"] / distro["n_celltypes"]
            x = np.sort(x)
            y = (1.0 - np.linspace(0, 1, len(x))) * len(x)
            ax.plot(x, y, label=tissue, color=colord[tissue])

        ax.grid(True)
        ax.set_ylabel("Number of genes")
        ax.set_title(species)
        if species != "z_mays":
            ax.set_ylim(0, 26000)
        else:
            ax.set_ylim(0, 40000)

        # Plot sigmoidal
        ngenes = distro["n_expressing"].shape[0]
        alpha, ymid = fit_alpha_sigmoidal(distrosi)
        y = np.linspace(0, 1, ngenes)
        x = 0.5 * (1.0 + erf(alpha * (y - ymid) / np.sqrt(2)))
        y = 1.0 - y
        y *= ngenes
        ax.plot(
            x,
            y,
            color="k",
            linestyle="--",
            label="$\\alpha = "
            + "{:.1f}".format(alpha)
            + "$\n$\\alpha_s = "
            + "{:.1e}".format(alpha / ngenes)
            + "$",
        )

        # Plot sigmoidal shade by tissue
        y = np.linspace(0, 1, ngenes)
        xs = []
        alphas = []
        for tissue in order:
            alpha, ymid = fit_alpha_sigmoidal(distrosi, tissues=[tissue])
            x = 0.5 * (1.0 + erf(alpha * (y - ymid) / np.sqrt(2)))
            xs.append(x)
            alphas.append(
                dict(tissue=tissue, alpha=alpha, alphas=alpha / ngenes, ngenes=ngenes)
            )
        alphas = pd.DataFrame(alphas)
        alphas["organism"] = species
        alphas_all.append(alphas)
        xs = np.array(xs)
        xmin = xs.min(axis=0)
        xmax = xs.max(axis=0)
        y = 1.0 - y
        y *= ngenes
        ax.fill_betweenx(y, xmin, xmax, color="k", alpha=0.3)

        ax.legend(
            ncol=2,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            bbox_transform=ax.transAxes,
            title="Species",
        )

    axs[-2].set_xlabel("Fraction of cell types expressing gene")
    axs[-1].set_xlabel("Fraction of cell types expressing gene")
    fig.tight_layout()

    alpha_all = pd.concat(alphas_all)

    species_multi = alpha_all["organism"].unique()
    colord = sns.color_palette("husl", n_colors=len(species_multi))
    colord = dict(zip(species_multi, colord))
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for species in species_multi:
        datum = alpha_all[alpha_all["organism"] == species]
        sns.kdeplot(datum["alphas"], ax=ax, label=species, color=colord[species])
    ax.legend(loc="upper right")
    ax.set_xlabel("Sigmoidal steepness $\\alpha_s$")
    ax.grid(True)
    fig.tight_layout()

    plt.ion()
    plt.show()
