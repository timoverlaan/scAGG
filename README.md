# scAGG: Sample-level embedding and classification of Alzheimer’s disease from single-nucleus data

The repository contains the code corresponding to the paper that is now available as preprint at: https://www.biorxiv.org/content/10.1101/2025.01.28.635240v1.abstract

Below, we provide instructions for reproducing the results presented in the paper.

## Abstract

Identifying key cell types and genes in Alzheimer’s Disease (AD) is crucial for understanding its pathogenesis and discovering therapeutic targets. Single cell RNA sequencing technology (scRNAseq) has provided unprecedented opportunities to study the molecular mechanisms that underlie AD at the cellular level. In this study, we address the problem of sample-level classification of AD using scRNAseq data, where we predict the disease status of entire samples from the gene expression profiles of their cells, which are not necessarily all affected by the disease. We introduce scAGG, a sample-level classification model which uses a sample-level pooling mechanism to aggregate single-cell embeddings, and show that it can accurately classify AD individuals and healthy controls. We then investigate the latent space learnt by the model and find that the model learns an ordering of the cells corresponding to disease severity. Genes associated with this ordering are enriched in AD-linked pathways, including cytokine signalling, apoptosis, and metal ion response. We also evaluate two attention-based models that perform on par with scAGG, but entropy analysis of their attention scores reveals limited interpretability value. As scRNAseq is increasingly applied to large cohorts, our approach provides a way to link individual phenotypes to single-cell measurements. Our cell- and sample-level severity scores may enable identification of AD-associated cell subtypes, paving the way for targeted drug development and personalized treatment strategies in AD.


## Setup

The recommended way to set up the environment is using [pixi](https://pixi.sh/latest/). After installing pixi, simply run `pixi install` to setup the dependencies. 

Alternatively, you may use any other environment management tool to install the dependencies listed int he `pixi.toml` file.

## Dataset processing

The dataset used in the paper is from the ROSMAP project, which was downloaded from the AD knowledge portal: https://www.synapse.org/Synapse:syn3219045. Before you can download the data from there, however, you have to request access: https://adknowledgeportal.synapse.org/Data%20Access.

The downloaded data was first combined into a single `.h5ad` file, to be used with the `scanpy` library for processing.

Given a `raw.h5ad` file, we used the following script and hyper-parameters to generate the dataset used to train/evaluate the model:

```sh
pixi run python src/pre_processing.py \
    --input data/raw.h5ad \
    --output data/dataset_1k.h5ad \
    --n_top_genes 1000 \
    --gene_selection seurat_v3 \
    --k_neighbors 30
```

For the experiments with 5000 genes, the corresponding parameter was updated above, and the script was run again.


### Using your own data

Alternatively, any other single-cell transcriptomics dataset may be used, as long as it's saved in `.h5ad` format, and contains the following columns in its `.obs`:

- "Label" (categorical, with either AD, CT, Other)
- "Donor ID" (categorical, some unique donor ID, that is the same for all cells of the same donor)
- "total_counts" (numerical, total number of counts of this donor, used for QC)
- "pct_counts_mt" (numerical, percentage of mitochondrial counts, used for QC)

## Training the model


### Evaluation of the baselines

## Reproducing analyses in the paper

