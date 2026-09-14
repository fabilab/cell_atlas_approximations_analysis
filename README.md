Cell atlas approximations - Analysis
=====================================
This repo contains the analysis of cell atlas approximations across organs and organisms to understand division of labour in multiceullar organisms.

The preprint is on bioRxiv:

https://www.biorxiv.org/content/10.1101/2025.02.19.639005v2.full

The scripts building the main figures in the manuscript are found within `atlasapprox_ana/saturn/analyse_output.py`.
Plain Python scripts were used in lieu of literate Jupyter notebooks for simplicity. That script also contains
the code defining phylogeny-adjusted pseudotime, factored out in a function at the top of the file, called `adjust_pseudotime_for_phylogeny`.
