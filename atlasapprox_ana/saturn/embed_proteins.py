"""Embed all proteins from all termite species using ESM.

This must be run inside the Python 3.9 esm conda environment:

source ~/miniconda3/bin/activate && conda activate esm
"""

import os
import pathlib
import subprocess as sp
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Embed all proteins from all species using ESM."
    )
    parser.add_argument("--species", default=None, help="Only process these species")
    parser.add_argument(
        "--model", default="esm1b", choices=["esm1b", "esmc", "esmc600"]
    )
    args = parser.parse_args()

    fasta_root_folder = pathlib.Path(
        "/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/peptide_sequences/"
    )
    fasta_files = os.listdir(fasta_root_folder)

    if args.model == "esm1b":
        output_root_folder = fasta_root_folder.parent / "esm_embeddings"
    elif args.model == "esmc":
        output_root_folder = fasta_root_folder.parent / "esmc_embeddings"
    else:
        output_root_folder = fasta_root_folder.parent / "esmc600_embeddings"

    os.makedirs(output_root_folder, exist_ok=True)

    for fasta_file in fasta_files:
        species = fasta_file.split(".")[0]
        if args.species is not None and species not in args.species:
            continue

        print(f"Processing {fasta_file}")

        fasta_file_abs_path = fasta_root_folder / fasta_file
        output_folder_abs_path = output_root_folder / f"{fasta_file}_{args.model}"
        if (args.model == "esm1b") and output_folder_abs_path.exists():
            print(f"Skipping {fasta_file}, already processed")
            continue
        else:
            os.makedirs(output_folder_abs_path, exist_ok=True)

        if args.model == "esm1b":
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

        else:
            from concurrent.futures import ThreadPoolExecutor
            from typing import Sequence, Union
            from Bio import SeqIO
            import torch
            from esm.models.esmc import ESMC
            from esm.sdk.api import (
                ESM3InferenceClient,
                ESMProtein,
                ESMProteinError,
                LogitsConfig,
                LogitsOutput,
                ProteinType,
            )

            EMBEDDING_CONFIG = LogitsConfig(sequence=True, return_embeddings=True)

            def embed_sequence(
                client: ESM3InferenceClient,
                sequence: str,
                fn_out: Union[str, pathlib.Path],
            ) -> LogitsOutput:
                protein = ESMProtein(sequence=sequence)
                protein_tensor = client.encode(protein)
                output = client.logits(protein_tensor, EMBEDDING_CONFIG)

                # Compute mean of embeddings across sequence length
                # https://github.com/evolutionaryscale/esm/blob/main/cookbook/tutorials/2_embed.ipynb
                emb_mean = output.embeddings.mean(axis=-2).squeeze()
                torch.save({"mean_representations": {-1: emb_mean}}, fn_out)

                return emb_mean

            model_name = "esmc_300m" if args.model == "esmc" else "esmc_600m"
            client = ESMC.from_pretrained(model_name).to("cuda")  # or "cpu"

            with open(fasta_file_abs_path, "rt") as handle:
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for ir, record in enumerate(SeqIO.parse(handle, "fasta")):
                        name = record.id
                        fn_out = output_folder_abs_path / f"{name}.pt"
                        # Exploit a trick for gene names with slashes here (e.g. frog)
                        os.makedirs(fn_out.parent, exist_ok=True)

                        if fn_out.exists():
                            print(f"{ir + 1} {name} already processed")
                            continue
                        print(ir + 1, name)

                        sequence = str(record.seq)
                        futures.append(
                            executor.submit(embed_sequence, client, sequence, fn_out)
                        )

                    for future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            print(ESMProteinError(500, str(e)))
