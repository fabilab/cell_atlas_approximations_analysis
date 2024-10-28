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

    def find_sister_type(
        source_organism,
        cell_type,
        target_organism,
        n_markers_original=30,
        n_markers_common_organs_original=15,
        max_homologs_per_query=5,
        n_cell_type_candidates=5,
        n_markers_target=30,
        n_markers_common_organs_target=10,
    ):
        api = atlasapprox.API()

        # Find which organs contain the original cell type
        celltypexorgans = api.celltypexorgan(organism=source_organism)
        organs_ct = celltypexorgans.loc[cell_type]
        organs_ct = organs_ct[organs_ct > 0].index

        # Find common markers for that cell type across organs
        markers = []
        for organ in organs_ct:
            markersi = api.markers(
                cell_type=cell_type,
                organism=source_organism,
                organ=organ,
                number=n_markers_original,
            )
            markersi = [
                {"organ": organ, "gene": x, "rank": i} for i, x in enumerate(markersi)
            ]
            markers.extend(markersi)
        markers = pd.DataFrame(markers)

        if len(organs_ct) > 1:
            markers_common_organs = (
                markers.groupby("gene")
                .size()
                .nlargest(n_markers_common_organs_original)
                .index
            )
        else:
            markers_common_organs = (
                pd.DataFrame(markersi)
                .iloc[:n_markers_common_organs_original]["gene"]
                .values
            )

        # Find homologs in target organism
        species = target_organism
        homologs = api.homologs(
            source_organism=source_organism,
            target_organism=target_organism,
            features=markers_common_organs,
        )
        # Limit to a reasonable number of homologs in target organism
        homologs = pd.concat(
            [
                group.nsmallest(max_homologs_per_query, "distances")
                for _, group in homologs.groupby("queries")
            ]
        )
        homologs_unique = homologs["targets"].unique()

        # Find cell types expressing those homologs across organs (candidates)
        organs_target = api.organs(organism=target_organism)
        cell_types_homo = []
        for organ in organs_target:
            frac = api.fraction_detected(
                organism=target_organism, organ=organ, features=homologs_unique
            )
            score = frac.sum(axis=0)
            cell_types_homoi = score.to_frame(name="score").reset_index()
            cell_types_homoi.rename(columns={"index": "cell_type"}, inplace=True)
            cell_types_homoi["organ"] = organ
            cell_types_homo.append(cell_types_homoi)
        cell_types_homo = pd.concat(cell_types_homo)
        cell_type_candidates = (
            cell_types_homo.groupby("cell_type")
            .score.sum()
            .sort_values(ascending=False)
            .index[:n_cell_type_candidates]
        )

        # Reverse search begins here: find markers for candidate cell types
        markers_candidates = []
        for organ in organs_target:
            celltypes = api.celltypes(organism=species, organ=organ)
            for ct_target in cell_type_candidates:
                if ct_target in celltypes:
                    markers_ct_target = api.markers(
                        organism=species,
                        organ=organ,
                        cell_type=ct_target,
                        number=n_markers_target,
                    )
                    markers_ct_target = [
                        {"organ": organ, "gene": x, "rank": i, "cell_type": ct_target}
                        for i, x in enumerate(markers_ct_target)
                    ]
                    markers_candidates.extend(markers_ct_target)
        markers_candidates = pd.DataFrame(markers_candidates)

        # Reverse homology search: how many of each candidate markers lead to common
        # markers of the original cell type?
        scores = []
        markers_common_organs_ext = (
            markers.groupby("gene")
            .size()
            .nlargest(n_markers_common_organs_original + 35)
            .index
        )
        for ct_target, markers_ct_target in markers_candidates.groupby("cell_type"):
            # Restrict to candidate markers common across target organism organs
            markers_ct_target = (
                markers_ct_target["gene"]
                .value_counts()[:n_markers_common_organs_target]
                .index
            )
            homologs_inverse = api.homologs(
                source_organism=species,
                target_organism="h_sapiens",
                features=markers_ct_target,
            )
            # Reverse score is how many of an extended list of common original markers
            # are hit by the reverse homology. This is by design inaccurate: exact
            # homology distance could mislead since the funcional homolog can be
            # not the closest match
            homologs_inverse["found"] = homologs_inverse["targets"].isin(
                markers_common_organs_ext
            )
            score = homologs_inverse["found"].sum()
            matching_markers = homologs_inverse.loc[
                homologs_inverse["found"], ["queries", "targets", "distances"]
            ]
            scores.append(
                {
                    "cell_type": ct_target,
                    "markers": markers_ct_target,
                    "score": score,
                    "matching_markers": tuple(matching_markers["queries"]),
                    "matching_markers_reverse_homologs": tuple(
                        matching_markers["targets"]
                    ),
                }
            )
        scores = (
            pd.DataFrame(scores)
            .set_index("cell_type")
            .sort_values("score", ascending=False)
        )
        # Find cell type among candidates with the highest reverse score
        matching_cell_type = scores["score"].idxmax()

        return {
            "best_match": matching_cell_type,
            "candidate_scores": scores,
        }

    # Start from a human cell type
    result = find_sister_type(
        source_organism="h_sapiens",
        cell_type="B",
        target_organism="m_musculus",
    )

    # Create bipartite graph of types
    print("Build bipartite graph")
    source_organism = "h_sapiens"
    target_organism = "m_musculus"
    n_cell_types = 50
    api = atlasapprox.API()
    celltypexorgans = api.celltypexorgan(organism=source_organism)
    source_types = (celltypexorgans > 0).sum(axis=1).nlargest(n_cell_types).index
    results = []
    for ct in source_types:
        print(ct)
        result = find_sister_type(
            source_organism=source_organism,
            cell_type=ct,
            target_organism=target_organism,
        )
        results.append(result)

    target_types = []
    edges = []
    for ct1, result in zip(source_types, results):
        tmp = result["candidate_scores"]
        tmp = tmp.loc[tmp["score"] > 1]
        for j, (ct2, row) in enumerate(tmp.iterrows()):
            if ct2 not in target_types:
                target_types.append(ct2)
            edges.append(
                {
                    "source": ct1,
                    "target": ct2,
                    "score": row["score"],
                    "best_match": j == 0,
                    "weight": 1.0 * row["score"] / tmp.iloc[0]["score"],
                }
            )
    edges = pd.DataFrame(edges)

    # Walk the bipartite to get optimal partnerings
    edges_optimal = []
    fwd_dic = {key: list(val.values) for key, val in edges.groupby("source").target}
    while True:
        found = False
        for source, targets in fwd_dic.items():
            if len(targets) == 1:
                found = True
                target = targets[0]
                print("unique", source, target)
                etmp = edges.groupby("source").get_group(source).iloc[0]
                edges_optimal.append(
                    {
                        "source": source,
                        "target": target,
                        "best_match": True,
                        "weight": etmp["weight"],
                    }
                )
                fwd_dic.pop(source)
                # exclude this target for anyone else, it's taken
                for source2, targets2 in fwd_dic.items():
                    # FIXME: need a better hack here
                    if len(targets2) == 1:
                        continue
                    if target in targets2:
                        fwd_dic[source2].remove(target)
                break
        if found:
            continue
        # end of obvious choices, take the closest matches now
        for source, targets in fwd_dic.items():
            found = True
            target = targets[0]
            print("closest", source, target)
            etmp = edges.groupby("source").get_group(source).iloc[0]
            best_match = etmp["target"] == target
            edges_optimal.append(
                {
                    "source": source,
                    "target": target,
                    "best_match": best_match,
                    "weight": etmp["weight"],
                }
            )
            fwd_dic.pop(source)
            # exclude this target for anyone else, it's taken
            for source2, targets2 in fwd_dic.items():
                if target in targets2:
                    fwd_dic[source2].remove(target)
            break
        if fwd_dic == {}:
            break
    edges_optimal = pd.DataFrame(edges_optimal)
    edges_optimal["order"] = [
        list(source_types).index(x) for x in edges_optimal["source"]
    ]
    edges_optimal = edges_optimal.sort_values("order")

    target_types_optimal = edges_optimal["target"].drop_duplicates().values

    fig, ax = plt.subplots()
    ax.scatter([0] * len(source_types), np.arange(len(source_types)), color="k")
    ax.scatter(
        [1] * len(target_types_optimal), np.arange(len(target_types_optimal)), color="k"
    )
    for i, ct1 in enumerate(source_types):
        ax.text(-0.1, i, ct1, ha="right", va="center")
    for i, ct2 in enumerate(target_types_optimal):
        ax.text(1.1, i, ct2, ha="left", va="center")
    for _, edge in edges_optimal.iterrows():
        ct1 = edge["source"]
        ct2 = edge["target"]
        i1 = list(source_types).index(ct1)
        i2 = list(target_types_optimal).index(ct2)
        ls = "-" if edge["best_match"] else "--"
        lw = 3.0 * edge["weight"]
        ax.plot([0, 1], [i1, i2], ls=ls, color="k", lw=lw, alpha=min(1.0, lw / 2.0))
    ax.set_xticks([0, 1])
    ax.set_xticklabels([source_organism, target_organism])
    ax.set_yticks([])
    ax.set_ylim(max(len(target_types_optimal), len(source_types)) - 0.5, -0.5)
    fig.tight_layout()

    plt.ion()
    plt.show()
