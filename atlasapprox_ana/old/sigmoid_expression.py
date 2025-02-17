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

try:
    import requests

    requests.get("http://127.0.0.1:5000/v1/data_sources")
    os.environ["ATLASAPPROX_BASEURL"] = "http://localhost:5000"
except requests.exceptions.ConnectionError:
    pass
finally:
    os.environ["ATLASAPPROX_HIDECREDITS"] = "yes"

import atlasapprox

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
        adata.var["n_expressing"] = n_celltypes_expressing
        adata.var["frac_expressing"] = 1.0 * n_celltypes_expressing / adata.shape[0]

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

    if False:
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

    if False:
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
        tissues = adata.obs["tissue"].unique()
        for tissue in tissues:
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
    species_order = sorted(
        distros_tissues.keys(),
        key=lambda x: len(distros_tissues[x]),
        reverse=True,
    )
    for isp, species in enumerate(species_order):
        ax = axs[min(isp, len(axs) - 1)]
        distrosi = distros_tissues[species]
        order = sorted(
            distrosi.keys(),
            key=lambda x: (distrosi[x]["n_expressing"] > 0.8).mean(),
            reverse=True,
        )
        colord = sns.color_palette("husl", n_colors=len(distrosi))
        colord = dict(zip(order, colord))

        for tissue in order:
            if isp < len(axs) - 1:
                color = colord[tissue]
            else:
                color = sns.color_palette("husl", n_colors=len(species_order) - 7)[
                    isp - 7
                ]
            distro = distrosi[tissue]
            x = 1.0 * distro["n_expressing"] / distro["n_celltypes"]
            x = np.sort(x)
            y = (1.0 - np.linspace(0, 1, len(x))) * len(x)
            ax.plot(
                x,
                y,
                label=tissue if isp < len(axs) - 1 else f"{species}/{tissue}",
                color=color,
            )

        ax.grid(True)
        ax.set_ylabel("Number of genes")
        if isp < len(axs) - 1:
            ax.set_title(species)
        else:
            ax.set_title("Other species")
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
        # if isp < len(axs) - 1:
        #    label = (
        #        "$\\alpha = "
        #        + "{:.1f}".format(alpha)
        #        + "$\n$\\alpha_s = "
        #        + "{:.1e}".format(alpha / ngenes)
        #        + "$"
        #    )
        # else:
        #    label = None
        ax.plot(
            x,
            y,
            color="k",
            linestyle="--",
            # label=label,
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
                dict(
                    tissue=tissue,
                    alpha=alpha,
                    alphas=alpha / ngenes,
                    ngenes=ngenes,
                    ymid=ymid,
                )
            )
        alphas = pd.DataFrame(alphas)
        alphas["organism"] = species
        alphas_all.append(alphas)
        xs = np.array(xs)
        xmin = xs.min(axis=0)
        xmax = xs.max(axis=0)
        y = 1.0 - y
        y *= ngenes
        if isp < len(axs) - 1:
            ax.fill_betweenx(y, xmin, xmax, color="k", alpha=0.3)

            ax.legend(
                ncol=2,
                loc="upper left",
                bbox_to_anchor=(1, 1),
                bbox_transform=ax.transAxes,
                title="Tissue",
            )

    axs[-1].legend(
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        bbox_transform=ax.transAxes,
        title="Species/tissue",
    )

    axs[-2].set_xlabel("Fraction of cell types expressing gene")
    axs[-1].set_xlabel("Fraction of cell types expressing gene")
    fig.tight_layout()

    alphas_all = pd.concat(alphas_all)

    # Alphas across species
    species_multi = alphas_all["organism"].unique()
    colord = sns.color_palette("husl", n_colors=len(species_multi))
    colord = dict(zip(species_multi, colord))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for species in species_multi:
        datum = alphas_all[alphas_all["organism"] == species]
        if len(datum) == 1:
            axs[0].axvline(
                1.0 / datum.iloc[0]["alpha"], label=species, color=colord[species]
            )
            axs[1].axvline(
                1.0 / datum.iloc[0]["alphas"], label=species, color=colord[species]
            )
        else:
            sns.kdeplot(
                1.0 / datum["alpha"], ax=axs[0], label=species, color=colord[species]
            )
            sns.kdeplot(
                1.0 / datum["alphas"], ax=axs[1], label=species, color=colord[species]
            )
    axs[0].set_xlabel("Sigmoidal inverse steepness $1 / \\alpha$")
    axs[1].set_xlabel("Rescaled inv steepness $ngenes / \\alpha_s$")
    axs[0].grid(True)
    axs[1].grid(True)
    axs[1].legend(
        loc="upper left", ncol=2, bbox_to_anchor=(1, 1), bbox_transform=axs[1].transAxes
    )
    fig.tight_layout()

    print("Plot alpha vs number of genes")
    species_multi = alphas_all["organism"].unique()
    colord = sns.color_palette("husl", n_colors=len(species_multi))
    colord = dict(zip(species_multi, colord))
    fig, ax = plt.subplots(1, 1, figsize=(7.7, 4.3))
    for species in species_multi:
        datum = alphas_all[alphas_all["organism"] == species]
        ax.scatter(
            datum["ngenes"],
            1.0 / datum["alpha"],
            label=species,
            color=colord[species],
            marker="s" if species in plants else "o",
        )
    ax.grid(True)
    ax.set_xlabel("Number of genes")
    ax.set_ylabel("Fraction of variable genes $1 / \\alpha$")
    ax.legend(
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(1, 1),
        bbox_transform=ax.transAxes,
        title="Species",
    )
    ax.set_xscale("log")
    ax2 = fig.add_axes([0.38, 0.72, 0.12, 0.2])
    from scipy.stats import gaussian_kde

    x = np.linspace(0, 0.3, 300)
    y1 = gaussian_kde(1.0 / alphas_all["alpha"].values, bw_method=0.4)(x)
    y2 = gaussian_kde(
        1.0 / alphas_all["alpha"].loc[alphas_all["organism"] == "h_sapiens"].values,
        bw_method=0.4,
    )(x)
    y1 /= y1.max()
    y2 /= y2.max()
    ax2.plot(x, y1, color="grey")
    ax2.fill_between(x, 0, y1, color="grey", alpha=0.5)
    ax2.plot(x, y2, color=colord["h_sapiens"])
    ax2.fill_between(x, 0, y2, color=colord["h_sapiens"], alpha=0.5)

    ax2.set_xlabel("$1 / \\alpha$")
    ax2.set_ylabel("")
    ax2.set_yticks([])
    ax2.set_xlim(0, 0.3)
    fig.tight_layout()

    plt.ion()
    plt.show()

    print("Plot ymid vs number of genes")
    alphas_all["ymids"] = alphas_all["ymid"] * alphas_all["ngenes"]
    species_multi = alphas_all["organism"].unique()
    colord = sns.color_palette("husl", n_colors=len(species_multi))
    colord = dict(zip(species_multi, colord))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)
    ax = axs[0]
    for species in species_multi:
        datum = alphas_all[alphas_all["organism"] == species]
        ax.scatter(
            datum["ngenes"],
            datum["ymids"],
            label=species,
            color=colord[species],
            marker="s" if species in plants else "o",
        )
    ax.grid(True)
    ax.set_xlabel("Number of genes")
    ax.set_ylabel("Number of expressed genes")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Plot guessed curve
    x = np.logspace(4, 5, 100)
    k = 2.0
    m0 = 2.0
    x0 = 2e4
    y0 = 2000

    def curve(x, x0, m0, y0, k):
        dx = np.log10(x) - np.log10(x0)
        dy = dx * (m0 - (1.0 / (1 + np.exp(-k * dx))))
        y = (10**dy) * y0
        return y

    from scipy.optimize import curve_fit

    pars = curve_fit(
        curve, alphas_all["ngenes"], alphas_all["ymids"], p0=[x0, m0, y0, k]
    )[0]
    y = curve(x, *pars)

    ax.plot(x, y, color="k", linestyle="--")
    ax.text(
        0.9,
        0.1,
        "$f(x) = y_0 \\: (x / x_0)  (m_0 - \\frac{1}{1 + e^{-k (x / x_0)}})}$",
        transform=ax.transAxes,
    )

    ax = axs[1]
    for species in species_multi:
        datum = alphas_all[alphas_all["organism"] == species]
        ax.scatter(
            datum["ngenes"],
            datum["ymid"],
            label=species,
            color=colord[species],
            marker="s" if species in plants else "o",
        )
    ax.grid(True)
    ax.set_xlabel("Number of genes")
    ax.set_ylabel("Fraction of expressed genes")
    ax.legend(
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(1, 1),
        bbox_transform=ax.transAxes,
        title="Species",
    )
    ax.set_xscale("log")
    ax2 = fig.add_axes([0.54, 0.25, 0.12, 0.2])
    from scipy.stats import gaussian_kde

    x = np.linspace(0, 0.95, 300)
    y1 = gaussian_kde(alphas_all["ymid"].values, bw_method=0.4)(x)
    y2 = gaussian_kde(
        alphas_all["ymid"].loc[alphas_all["organism"] == "h_sapiens"].values,
        bw_method=0.4,
    )(x)
    y1 /= y1.max()
    y2 /= y2.max()
    ax2.plot(x, y1, color="grey")
    ax2.fill_between(x, 0, y1, color="grey", alpha=0.5)
    ax2.plot(x, y2, color=colord["h_sapiens"])
    ax2.fill_between(x, 0, y2, color=colord["h_sapiens"], alpha=0.5)

    ax2.set_xlabel("Fraction")
    ax2.set_ylabel("")
    ax2.set_yticks([])
    ax2.set_xlim(0, 0.95)
    fig.tight_layout()

    plt.ion()
    plt.show()

    # Check numbers of paralogs along the sigmoidal
    def get_paralogs(species, max_distance_over_min=40, ncands=50):
        adata = adata_dict[species]
        api = atlasapprox.API()
        dists = []
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
            cands = adata.var.query(
                f"{binl} <= frac_expressing < {binr}"
            ).index.values.copy()
            np.random.shuffle(cands)
            cands = cands[:ncands]

            dis = api.homologs(
                source_organism=species,
                target_organism=species,
                features=cands,
                max_distance_over_min=max_distance_over_min,
            )
            dis = dis.groupby("queries").size()
            dists.append(dis)

        return dists

    species = "h_sapiens"
    adata = adata_dict[species]
    dists = get_paralogs(species, ncands=70)

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
    colors = sns.color_palette("husl", n_colors=len(dists))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, (binl, binr) in enumerate(bins):
        dis = dists[i]
        x = np.sort(dis.values)
        y = 1.0 - np.linspace(0, 1, len(x))
        ax.plot(x, y, label=f"{binl} <= frac < {binr}", color=colors[i])
    ax.set_xlabel("Number of paralogs")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        bbox_transform=ax.transAxes,
        title="Fraction of cell types expressing gene",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)
    fig.tight_layout()

    plt.ion()
    plt.show()

    # Ask how many members of a paralog family are expressed how much
    binl, binr = 0.95, 1.01
    ncands = 100
    max_distance_over_min = 40
    api = atlasapprox.API()
    cands = adata.var.query(f"{binl} <= frac_expressing < {binr}").index.values.copy()

    frac_avg = pd.Series(
        adata[:, cands].layers["fraction"].mean(axis=0),
        index=cands,
    )

    cands2 = (
        frac_avg[
            ~(
                frac_avg.index.str.startswith("MT-")
                | frac_avg.index.str.startswith("RPL")
                | frac_avg.index.str.startswith("RPS")
                | frac_avg.index.str.startswith("HLA-")
            )
        ]
        .nlargest(ncands)
        .index
    )

    dis = api.homologs(
        source_organism=species,
        target_organism=species,
        features=cands2,
        max_distance_over_min=max_distance_over_min,
    )

    tmp = dis.groupby("queries").size()
    dis_para = dis.loc[dis["queries"].isin(tmp.index[tmp > 1])]

    cands3 = dis_para.groupby("queries").size().sort_values(ascending=False).index

    def plot_paralogs(gene, ax=None, tight_layout=True, add_names=False):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        else:
            fig = ax.figure

        paralogs_rows = dis_para.loc[dis_para["queries"] == gene]
        paralog_dis = paralogs_rows["distances"].values
        paralogs = paralogs_rows["targets"].values
        frac_exp = adata.var.loc[list(paralogs), "frac_expressing"]
        s = [70 if x == gene else 10 for x in paralogs]
        c = ["r" if x == gene else "k" for x in paralogs]

        sc = ax.scatter(paralog_dis, frac_exp, s=s, c=c, zorder=5)
        if add_names:
            txts = [
                ax.text(x, y, paralog, va="center", ha="center")
                for x, y, paralog in zip(paralog_dis, frac_exp, paralogs)
            ]
        ax.grid(True)
        ax.set_xlabel("PROST distance")
        ax.set_ylabel("Fraction of cell types\nexpressing gene")
        ax.set_title(gene)
        if tight_layout:
            fig.tight_layout()

        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(-10, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", alpha=0.4),
            arrowprops=dict(arrowstyle="->"),
            clip_on=False,
        )
        annot.set_visible(False)
        names = paralogs

        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = " ".join([names[n] for n in ind["ind"]])
            annot.set_text(text)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        return ax

    nrows, ncols = 5, 5
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows), sharex=True, sharey=True
    )
    axs = axs.ravel()
    for i, gene in enumerate(cands3[: nrows * ncols]):
        plot_paralogs(gene, ax=axs[i], tight_layout=False)
        if i % ncols:
            axs[i].set_ylabel("")
        if i < len(axs) - ncols:
            axs[i].set_xlabel("")
    fig.tight_layout()

    # Frog had a whole genome duplication: study how that affects the paralogs
    species = "x_laevis"
    adata = adata_dict[species]

    genes_L = adata.var_names[adata.var_names.str.endswith(".L")]
    genes_S = adata.var_names[adata.var_names.str.endswith(".S")]
    tmpL = [x[:-2] for x in genes_L]
    tmpS = [x[:-2] for x in genes_S]
    tmp_both = list(set(tmpL) & set(tmpS))
    genes_both = np.array([[x + ".L", x + ".S"] for x in tmp_both])

    frac_comp = pd.DataFrame([], index=tmp_both)
    frac_comp["L"] = adata.var.loc[genes_both[:, 0]]["frac_expressing"].values

    frac_comp["S"] = adata.var.loc[genes_both[:, 1]]["frac_expressing"].values
    frac_comp["feature_L"] = [x + ".L" for x in frac_comp.index]
    frac_comp["feature_S"] = [x + ".S" for x in frac_comp.index]

    result = []
    for i in range(len(frac_comp) // 50):
        print(i)
        feas1 = frac_comp["feature_L"].iloc[i * 50 : (i + 1) * 50]
        feas2 = frac_comp["feature_S"].iloc[i * 50 : (i + 1) * 50]
        res = api.homology_distances(
            source_organism=species,
            target_organism=species,
            source_features=feas1,
            target_features=feas2,
        )
        res.index = [x[:-2] for x in res["queries"].values]
        result.append(res)
    result = pd.concat(result)
    frac_comp = frac_comp.loc[result.index]
    frac_comp["distances"] = result["distances"].values
    frac_comp["delta_frac"] = frac_comp["S"] - frac_comp["L"]

    if False:
        # Scatter distance vs delta fraction, supplementary
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(frac_comp["distances"], np.abs(frac_comp["delta_frac"]), s=10)
        ax.grid(True)
        ax.set_xlabel("PROST distance")
        ax.set_ylabel("Abs difference in fraction of\ncell types expressing gene")
        fig.tight_layout()

    print("Correlate PROST distance with delta fraction")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = [
        [0, 10],
        [5, 15],
        [10, 20],
        [15, 25],
        [20, 30],
        [25, 35],
        [30, 40],
        [35, 45],
        [40, 50],
        [45, 55],
        [50, 60],
        [55, 65],
        [60, 70],
    ]
    colors = sns.color_palette("rainbow", n_colors=len(bins))
    avgs = []
    for i, color in enumerate(colors):
        binl, binr = bins[i]
        frac_comp_bin = frac_comp.query(f"{binl} <= distances < {binr}")
        x = np.sort(np.abs(frac_comp_bin["delta_frac"].values))
        avg = x.mean()
        avgs.append(avg)
        y = 1.0 - np.linspace(0, 1, len(x))
        binc = 0.5 * (binl + binr)
        ax.plot(x, y, label=f"$d_P \\approx {binc}$", color=color, zorder=5)
    ax.set_xlabel("Abs difference in fraction of\ncell types expressing gene")
    ax.set_ylabel("Fraction of genes with $\\vert\\Delta f\\vert > x$")
    ax.legend(
        loc="upper left",
        title="PROST distance:",
        ncols=1,
        bbox_to_anchor=(1, 1),
        bbox_transform=ax.transAxes,
    )
    ax.grid(True)

    bincs = [0.5 * (binl + binr) for binl, binr in bins]
    axin = fig.add_axes([0.43, 0.64, 0.3, 0.3])
    axin.plot(bincs, avgs, color="k")
    xfit = np.linspace(0, 70, 100)
    yfit = 0.1 + 0.03 * xfit + 10 * xfit**2
    axin.plot(xfit, yfit, color="tomato", linestyle="--")
    axin.set_xlabel("PROST distance")
    axin.set_ylabel("$\\left< \\vert\\Delta f\\vert \\right>$")
    axin.grid(True)
    axin.set_xlim(left=0)
    axin.set_ylim(bottom=0)
    fig.tight_layout()
