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

blacklist_species = []
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
        min_candidate_score=0.3,
        n_markers_target=30,
        n_markers_common_organs_target=10,
        check_reverse=True,
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
        homologs["order"] = homologs["queries"].map(
            lambda x: list(markers_common_organs).index(x)
        )

        # Get the unique list of homologs, sorted as faithfully as possible
        homologs_unique = homologs.groupby("targets").order.min().sort_values().index

        # Find cell types expressing those homologs across organs (candidates)
        organs_target = api.organs(organism=target_organism)
        cell_types_homo = []
        for organ in organs_target:
            frac = api.fraction_detected(
                organism=target_organism, organ=organ, features=homologs_unique
            )
            # avg = api.average(
            #    organism=target_organism, organ=organ, features=homologs_unique
            # )
            score = frac.sum(axis=0)
            # if "pericyte" in score.index:
            #    __import__("ipdb").set_trace()
            cell_types_homoi = score.to_frame(name="score").reset_index()
            cell_types_homoi.rename(columns={"index": "cell_type"}, inplace=True)
            cell_types_homoi["organ"] = organ
            cell_types_homo.append(cell_types_homoi)
        cell_types_homo = pd.concat(cell_types_homo)

        # FIXME: sum() strongly biases toward common cell types, one has to
        # improve. One argument is to check what organs the source cell type
        # is found in and match.
        # mean() is subject to weird effects by which suboptimal matches due
        # to misannoations etc. drag down the entire score.
        cell_type_candidate_scores = (
            cell_types_homo.groupby("cell_type")
            .score.max()
            .nlargest(n_cell_type_candidates)
        )

        # __import__("ipdb").set_trace()

        # Normalise scores to discount the number of organs sampled
        cell_type_candidate_scores /= cell_type_candidate_scores.max()

        idx = cell_type_candidate_scores >= min_candidate_score
        cell_type_candidate_scores = cell_type_candidate_scores.loc[idx]

        if not check_reverse:
            return cell_type_candidate_scores

        cell_type_candidate_scores = cell_type_candidate_scores.to_frame(name="fwd")

        # Compute reverse scores using recursion
        cell_type_candidate_scores["rev"] = 0.0
        results_inverse = {}
        for ct_target in cell_type_candidate_scores.index:
            inverse_candidate_scores = find_sister_type(
                source_organism=target_organism,
                target_organism=source_organism,
                cell_type=ct_target,
                check_reverse=False,
            )
            results_inverse[ct_target] = inverse_candidate_scores
            cell_type_candidate_scores.at[ct_target, "rev"] = (
                inverse_candidate_scores.to_dict().get(cell_type, 0)
                / inverse_candidate_scores.max()
            )

        # Weight fwd and reverse scores equally, and use L0.5 distance
        # to bias such that the winner is not found only in one direction
        cell_type_candidate_scores["compound"] = np.sqrt(
            cell_type_candidate_scores[["fwd", "rev"]]
        ).mean(axis=1)
        cell_type_candidate_scores = cell_type_candidate_scores.sort_values(
            "compound", ascending=False
        )

        idx = cell_type_candidate_scores["compound"] >= min_candidate_score
        cell_type_candidate_scores = cell_type_candidate_scores.loc[idx]

        # Add a bunch of metadata to the result
        scores = (
            cell_type_candidate_scores[["compound"]]
            .reset_index()
            .rename(columns={"compound": "score", "cell_type": "cell_type_target"})
        )
        scores["source_organism"] = source_organism
        scores["target_organism"] = target_organism
        scores["cell_type_source"] = cell_type
        matching_cell_type = scores.at[scores["score"].idxmax(), "cell_type_target"]

        return {
            "best_match": matching_cell_type,
            "candidate_scores": scores,
        }

    if False:
        # Start from a human cell type
        result = find_sister_type(
            source_organism="h_sapiens",
            cell_type="pericyte",
            target_organism="m_murinus",
        )

        result_rev = find_sister_type(
            source_organism="m_musculus",
            cell_type=result["best_match"],
            target_organism="h_sapiens",
        )

    if True:
        # Create bipartite graph of types
        print("Build bipartite graph")
        source_organism = "h_sapiens"
        target_organism = "m_musculus"
        api = atlasapprox.API()
        # n_cell_types = 30
        # celltypexorgans = api.celltypexorgan(organism=source_organism)
        # source_types = (celltypexorgans > 0).sum(axis=1).nlargest(n_cell_types).index
        source_types = [
            "T",
            "NK",
            "B",
            "plasma",
            "macrophage",
            "monocyte",
            "neutrophil",
            # "mast",
            # "basophil",
            "dendritic",
            "plasmacytoid",
            "basal",
            "ciliated",
            "AT2",
            "urothelial",
            "epithelial",
            "alpha",
            "beta",
            "PP",
            "hepatocyte",
            "fibroblast",
            "smooth muscle",
            "pericyte",
            # "vascular smooth muscle",
            "venous",
            "arterial",
            "capillary",
            "lymphatic",
        ]
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
            tmp = tmp.loc[tmp["score"] > 0.2]
            for j, (_, row) in enumerate(tmp.iterrows()):
                ct2 = row["cell_type_target"]
                if ct2 not in target_types:
                    target_types.append(ct2)
                edges.append(
                    {
                        "source": ct1,
                        "target": ct2,
                        "score": row["score"],
                        "best_match": _ == tmp["score"].idxmax(),
                        "weight": 1.0 * row["score"] / tmp.iloc[0]["score"],
                    }
                )
        edges = pd.DataFrame(edges)

        edges_best = edges[edges["best_match"]]
        target_types_best = edges_best["target"].drop_duplicates().values

        super_types = [
            (
                "immune",
                [
                    "T",
                    "NK",
                    "B",
                    "plasma",
                    "macrophage",
                    "alveolar macrophage",
                    "monocyte",
                    "neutrophil",
                    "mast",
                    "basophil",
                    "eosinophil",
                    "plasmacytoid",
                    "dendritic",
                ],
            ),
            (
                "epithelial",
                [
                    "basal",
                    "ciliated",
                    "AT2",
                    "urothelial",
                    "epithelial",
                    "keratinocyte",
                ],
            ),
            (
                "endocrine",
                ["beta", "alpha", "PP", "enteroendocrine", "delta", "hepatocyte"],
            ),
            ("endothelial", ["capillary", "venous", "arterial", "lymphatic"]),
            (
                "mesenchymal",
                [
                    "fibroblast",
                    "smooth muscle",
                    "vascular smooth muscle",
                    "mesangial",
                    "pericyte",
                ],
            ),
            ("other", ["spermatocyte"]),
        ]
        dot_supercolord = {
            "immune": "navy",
            "epithelial": "tomato",
            "endocrine": "deeppink",
            "endothelial": "blueviolet",
            "mesenchymal": "orange",
            "other": "gray",
        }
        dot_colord = {}
        for super_type, types in super_types:
            for ct in types:
                dot_colord[ct] = dot_supercolord[super_type]
        colors1 = [dot_colord[x] for x in source_types]
        colors2 = [dot_colord.get(x, "black") for x in target_types_best]
        fig, ax = plt.subplots()
        ax.scatter(
            [0] * len(source_types), np.arange(len(source_types)), c=colors1, zorder=5
        )
        ax.scatter(
            [1] * len(target_types_best),
            np.arange(len(target_types_best)),
            c=colors2,
            zorder=5,
        )
        for i, ct1 in enumerate(source_types):
            ax.text(-0.1, i, ct1, ha="right", va="center")
        for i, ct2 in enumerate(target_types_best):
            ax.text(1.1, i, ct2, ha="left", va="center")
        for _, edge in edges_best.iterrows():
            ct1 = edge["source"]
            ct2 = edge["target"]
            i1 = list(source_types).index(ct1)
            i2 = list(target_types_best).index(ct2)
            lw = 2.0 * edge["weight"]
            if dot_colord[ct1] == dot_colord.get(ct2, "black"):
                color = dot_colord[ct1]
            else:
                color = "k"
            if ct1 == ct2:
                ls = "-"
            else:
                ls = "--"
            ax.plot(
                [0, 1], [i1, i2], ls=ls, color=color, lw=lw, alpha=min(1.0, lw / 2.0)
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([source_organism, target_organism])
        ax.set_yticks([])
        ax.set_ylim(max(len(target_types_best), len(source_types)) - 0.5, -0.5)
        fig.tight_layout()

        plt.ion()
        plt.show()

    if False:
        # Find remote sister cell types for muscle
        organisms = [
            "m_murinus",
            "x_laevis",
            "h_sapiens",
            "d_melanogaster",
            "m_musculus",
            "c_gigas",
            "c_hemisphaerica",
            "c_intestinalis",
            "s_pistillata",
            "a_queenslandica",
            "c_elegans",
            "d_rerio",
            "h_miamia",
            "i_pulchra",
            "m_leidyi",
            "n_vectensis",
            "p_crozieri",
            "s_mansoni",
            "s_mediterranea",
            "s_lacustris",
            "t_adhaerens",
            "s_purpuratus",
            "p_dumerilii",
            "h_vulgaris",
        ]
        api = atlasapprox.API()
        results = {}
        for organism in organisms:
            for organism2 in organisms:
                print(organism, organism2)
                cell_types = api.celltypexorgan(organism=organism).index
                cell_types_muscle = [
                    ct for ct in cell_types if "muscle" in ct or "cardiomyo" in ct
                ]
                if len(cell_types_muscle) == 0:
                    continue
                try:
                    result = find_sister_type(
                        source_organism=organism,
                        target_organism=organism2,
                        cell_type=cell_types_muscle,
                    )
                    results[organism] = result
                except:
                    pass
        results = pd.concat([x["candidate_scores"] for x in results.values()])
