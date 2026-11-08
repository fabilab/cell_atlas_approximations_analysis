"""Embed all proteins from Zootermopsis nevadensis using ESM1b, for the paper revision.

This must be run inside the Python 3.9 esm conda environment:

source ~/miniconda3/bin/activate && conda activate esm

(e.g. on the archfabilab1 workstation)
"""

import os
import glob
import pathlib
import subprocess as sp
import argparse
import torch
from tqdm import tqdm
from collections import defaultdict


# Last layer of pretrained transformer
LAST_LAYER = 33  # ESM1b
MSA_LAST_LAYER = 12  # MSA
LAST_LAYER_2 = 48  # ESM2
LAST_LAYER_C = -1  # ESMc


species = "Znev"
embedding_model = "ESM1b"


def summarize_gene_embeddings(input_fdn, output_fdn) -> None:
    """Convert protein embeddings to gene embeddings by averaging the protein embeddings for each gene."""

    embedding_model_lower = embedding_model.lower()
    output_fn = output_fdn / f"{species}_gene_all_{embedding_model_lower}.pt"

    print(f"Summarizing gene embeddings for {species} using {embedding_model}...")
    print("Input folder:", input_fdn)
    print("Output file:", output_fn)

    # Get last layer
    if embedding_model == "ESM1b":
        last_layer = LAST_LAYER
    elif embedding_model == "MSA1b":
        last_layer = MSA_LAST_LAYER
    elif embedding_model == "ESM2":
        last_layer = LAST_LAYER_2
    elif embedding_model == "ESMc":
        last_layer = LAST_LAYER_C
    elif embedding_model == "ESMc600":
        last_layer = LAST_LAYER_C
    else:
        raise ValueError(f'Embedding model "{embedding_model}" is not supported.')

    # Get protein embedding paths
    protein_embedding_paths = glob.glob(str(input_fdn) + "/*.pt")

    # Create mapping from gene name to embedding, considering the proteins are already representatives
    # NOTE: This differs from the original SATURN implementation, which averages the embeddings of all isoforms
    # within each gene, with equal weights across isof
    gene_symbol_to_embedding = defaultdict(list)
    for protein_embedding_path in tqdm(protein_embedding_paths):
        isoform_name = pathlib.Path(protein_embedding_path).stem
        # Isoform names are gene-R0
        gene = isoform_name[: isoform_name.rfind("-")]

        tmp = torch.load(protein_embedding_path)
        embedding = tmp["mean_representations"][last_layer]

        gene_symbol_to_embedding[gene].append(embedding)

    genes = list(gene_symbol_to_embedding.keys())
    print(genes[:10])

    print("Average embeddings for each gene")
    for gene, embeddings_isoforms in tqdm(gene_symbol_to_embedding.items()):
        gene_symbol_to_embedding[gene] = torch.mean(
            torch.vstack(embeddings_isoforms), dim=0
        )

    print("Save embedding summary dict to file:")
    torch.save(gene_symbol_to_embedding, output_fn)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and/or summarise gene embeddings for Zootermopsis."
    )
    parser.add_argument(
        "--summarise-only", action="store_true", help="Only summarise gene embeddings"
    )

    args = parser.parse_args()

    data_root_folder = pathlib.Path(
        "/home/fabio/projects/compressed_atlas/cell_atlas_approximations_analysis/data/znev"
    )
    fasta_file_abs_path = (
        data_root_folder / "peptide_sequences" / "znev_peptides_all_isoforms.fasta"
    )
    output_folder_abs_path = data_root_folder / "esm1b_embeddings"
    os.makedirs(output_folder_abs_path, exist_ok=True)

    if not args.summarise_only:
        script_path = (
            pathlib.Path("/home/fabio/projects/termites")
            / "software"
            / "esm"
            / "scripts"
            / "extract.py"
        )
        call = [
            "python",
            str(script_path),
            "esm1b_t33_650M_UR50S",
            str(fasta_file_abs_path),
            str(output_folder_abs_path),
            "--include",
            "mean",
        ]
        print(" ".join(call))
        sp.run(" ".join(call), check=True, shell=True)

    output_summary_path = data_root_folder / "esm1b_embeddings_summaries"
    os.makedirs(output_summary_path, exist_ok=True)
    summarize_gene_embeddings(output_folder_abs_path, output_summary_path)
