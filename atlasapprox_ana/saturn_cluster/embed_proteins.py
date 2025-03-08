"""Embed all proteins from all termite species using ESM.

This must be run inside the Python 3.9 esm conda environment:

source ~/miniconda3/bin/activate && conda activate esm

or, for esmc, inside the Python (3.12 seems ok) conda environment:

source ~/miniforge3/bin/activate && conda activate esmc

"""

import os
import pathlib
import subprocess as sp
import argparse


def run_esm1b(
    fasta_file_abs_path,
    output_folder_abs_path,
    ):
    root_fdn = pathlib.Path("/home/fabio/projects/termites")
    if not root_fdn.exists():
        root_fdn = pathlib.Path("/srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis")
    script_path = (
        root_fdn
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


def run_esmc(
    fasta_file_abs_path,
    output_folder_abs_path,
    model,
    nthreads=1,
):
    """Run ESM Cambrian."""
    from Bio import SeqIO
    import torch
    from esm.models.esmc import ESMC
    from esm.sdk.api import (
        ESMProtein,
        LogitsConfig,
        ESMProteinError,
    )

    # Open client once only
    # The model ends up in ~/.cache/huggingface/hub/...
    if model == "esmc600":
        model_name = "esmc_600m"
    else:
        model_name = "esmc_300m"

    embedding_config = LogitsConfig(
        sequence=True,
        return_embeddings=True,
    )

    def fun(client, name, sequence):
        """Embedding function for mapping, multithread, etc."""
        protein = ESMProtein(sequence=sequence)
        protein_tensor = client.encode(protein)
        # NOTE: we might want a specific hidden layer, like PROST does on ESM1b?
        logits_output = client.logits(
            protein_tensor,
            embedding_config,
        )

        # Compute mean of embeddings across sequence length
        # https://github.com/evolutionaryscale/esm/blob/main/cookbook/tutorials/2_embed.ipynb
        emb_mean = logits_output.embeddings.mean(axis=-2).squeeze()

        # Store to file
        fn_out = output_folder_abs_path / f"{name}.pt"
        torch.save({"mean_representations": {-1: emb_mean}}, fn_out)

    print("Load sequences from FASTA")
    names, sequences = [], []
    with open(fasta_file_abs_path, "rt") as handle:
        for ir, record in enumerate(SeqIO.parse(handle, "fasta")):
            name = record.id
            if (ir % 100) == 0:
                print(ir + 1, name)

            sequence = str(record.seq)
            if len(sequence) == 0:
                continue

            fn_out = output_folder_abs_path / f"{name}.pt"
            if fn_out.exists():
                #print(f"{ir + 1} {name} already processed")
                continue

            # Exploit a trick for gene names with slashes here (e.g. frog)
            os.makedirs(fn_out.parent, exist_ok=True)

            names.append(name)
            sequences.append(sequence)

    if len(names) == 0:
        print("All peptides for this species already processed")
        return
    
    # NOTE: empirical tests show that the client is not thread-safe. It tends to trip when too many requests are made
    # It's probably possible to create a pool of clients (each with a copy of the model) and run those in parallel
    # Basically, one would need to instantiate say 5 copies of the model, then make a threadpoolexecutor with 5 workers,
    # then load a certain model based on what worker we are on. Each worker will process one sequence at a time only,
    # therefore this should be safe. So how do we figure out what worker we are on?
    if nthreads > 1:
        from concurrent.futures import ThreadPoolExecutor
        import threading

        # We need to create a dict of clients to ensure the same client is not under pressure from multiple requests
        # at the same time. By definition each thread will be running in series, therefore it will only source its
        # own client
        clientd = {}
        def initialiser(self):
            clientd[threading.current_thread().name] = ESMC.from_pretrained(model_name).to("cuda")

        def runner(self, *args):
            client = clientd[threading.current_thread().name]
            return fun(client, *args)

        print("Begin multithreading")
        with ThreadPoolExecutor(max_workers=nthreads, initializer=initialiser) as executor:
            print("Queue promises")
            futures = [executor.submit(runner, name, sequence) for name, sequence in zip(names, sequences)]

            nfutures = len(futures)
            print(f"Fulfill {nfutures} promises")
            for name, sequence, future in zip(names, sequences, futures):
                try:
                    future.result()
                except Exception as e:
                    print(name)
                    print(sequence)
                    print(ESMProteinError(500, str(e)))
    else:
        print("Use single thread for safety")
        print(f"Load model: {model_name}")
        client = ESMC.from_pretrained(model_name).to("cuda")
        npeptides = len(names)
        print(f"Embed {npeptides} peptide sequences")
        for name, sequence in zip(names, sequences):
            print(f"  Embed {name}: {sequence}")
            fun(client, name, sequence)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Embed all proteins from all species using ESM."
    )
    parser.add_argument("--species", default=None, help="Only process these species")
    parser.add_argument("--model", default="esm1b", choices=["esm1b", "esmc", "esmc600"])
    parser.add_argument("--species-number", type=int, default=None, help="Process only one species")
    parser.add_argument("--nthreads", type=int, default=1, help="Multithread embedding to exploit the full GPU memory")
    args = parser.parse_args()

    fasta_root_folder = pathlib.Path(
        "/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/peptide_sequences/"
    )
    if not fasta_root_folder.exists():
        fasta_root_folder = pathlib.Path(
            "/srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/data/reference_atlases/peptide_sequences/"
        )

    # Species without an atlas are stored elsewhere to limit interactions
    if (args.species is not None):
        fasta_noatlas_root_folder = pathlib.Path(
            "/srv/scratch/fabilab/fabio/projects/cell_atlas_approximations_analysis/data/noatlas_species/peptide_sequences/"
        )
        noatlas_fns = os.listdir(fasta_noatlas_root_folder)
        if f"{args.species}.fasta" in noatlas_fns:
            fasta_root_folder = fasta_noatlas_root_folder

    fasta_files = os.listdir(fasta_root_folder)
    
    if args.model == "esm1b":
        output_root_folder = fasta_root_folder.parent / "esm_embeddings"
    elif args.model == "esmc600":
        output_root_folder = fasta_root_folder.parent / "esmc600_embeddings"
    else:
        output_root_folder = fasta_root_folder.parent / "esmc_embeddings"
    os.makedirs(output_root_folder, exist_ok=True)

    for isn, fasta_file in enumerate(fasta_files):
        if (args.species_number is not None) and (args.species_number != (isn + 1)):
            continue

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

        # NOTE: if on katana, move the fasta file to local scratch
        # of the computing node to speed access in the loop
        local_scratch = os.getenv("TMPDIR")
        if local_scratch is not None:
            print("Copying fasta file onto local hard drive")
            import shutil
            fasta_file_abs_path_local = pathlib.Path(local_scratch) / f"embed_proteins_{species}_peptides.fasta"
            shutil.copy(
                fasta_file_abs_path,
                fasta_file_abs_path_local,
            )
            fasta_file_abs_path = fasta_file_abs_path_local

        if args.model == "esm1b":
            run_esm1b(fasta_file_abs_path, output_folder_abs_path)
        else:
            run_esmc(fasta_file_abs_path, output_folder_abs_path, args.model, 
                     nthreads=args.nthreads)
