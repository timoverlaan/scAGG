# scAGG: Sample-level embedding and classification of Alzheimer’s disease from single-nucleus data

The repository contains the code corresponding to the paper that is now available as preprint at: https://www.biorxiv.org/content/10.1101/2025.01.28.635240v1.abstract

Below, we provide instructions for reproducing the results presented in the paper.

## Setup

The recommended way to set up the environment is using [pixi](https://pixi.sh/latest/). After installing pixi, simply run `pixi install` to setup the dependencies. 

Alternatively, you may use any other environment management tool to install the dependencies listed int he `pixi.toml` file.

## Dataset processing

The dataset used in the paper is from the ROSMAP project, which was downloaded from the AD knowledge portal: https://www.synapse.org/Synapse:syn3219045. Before you can download the data from there, however, you have to request access: https://adknowledgeportal.synapse.org/Data%20Access.

The downloaded data was first combined into a single `.h5ad` file, to be used with the `scanpy` library for processing.

Given a `raw.h5ad` file, we used the following script and hyper-parameters to generate the dataset used to train/evaluate the model:

```sh
pixi run python src/pre_processing.py --input data/raw.h5ad --output data/dataset_1k.h5ad --n_top_genes 1000 --gene_selection seurat_v3 --k_neighbors 30
```

For the experiments with 5000 genes, the corresponding parameter was updated above, and the script was run again.


### Using your own data

Alternatively, any other single-cell transcriptomics dataset may be used, as long as it's saved in `.h5ad` format, and contains the following columns in its `.obs`:

- "Label" (categorical, with either AD, CT, Other)
- "Donor ID" (categorical, some unique donor ID, that is the same for all cells of the same donor)
- "total_counts" (numerical, total number of counts of this donor, used for QC)
- "pct_counts_mt" (numerical, percentage of mitochondrial counts, used for QC)

## Training the model

The model is trained using the `train.py` script. We used the following command and hyper-parameters for base scAGG:
```sh
pixi run python src/train.py --dataset data/dataset_1k.h5ad --n-epochs 2 --dim 32 --split-seed 42 --batch-size 8 --dropout 0.1 --learning-rate 0.001 --pooling mean --label wang --n_splits 5 --no-graph
```

And the following HPs were used for the GAT-based model:
```sh
pixi run python src/train.py --dataset data/dataset_1k.h5ad --dim 16 --split-seed 42 --batch-size 8 --dropout 0.5 --learning-rate 0.001 --pooling mean --label wang --n_splits 5  --n-epochs 5 --save-embeddings --save-attention --save
```

To enable attention-based pooling in either of the models, the `--pooling mean` parameter was changed to `--pooling self-att-sum`.

After running the script above, a new time-stamped `.h5ad` file will be output in the `out/` folder, containing information about the performance, embeddings and attention scores.


### Evaluation of the baselines

The other baseline models that we compare with in the paper are can be run using their corresponding notebooks and scripts in the `src/models/baselines` directory.

## Reproducing analyses in the paper

We now give an overview of the notebooks and scripts required to reproduce the results presented in the paper:

- **Figure 2**: Shows performance results from execution of experiments described above.
- **Figure 3**: These visualizations are made using the notebook `src/analysis/fig_sample_embeddings.ipynb`
- **Figure 4**: The UMAPs are calculated using the `src/analysis/fig_umap.ipynb` notebook. And the LISI scores are calculated using the `run_lisi.py` script.
- **Figure 5**: The two plots can be reproduced using the `src/analysis/extreme_cells_emb.ipynb` notebook. The GSEA figure (panel c) is made with the output of `src/analysis/gsea.py`, to make the figure itself in `src/analysis/fig_gsea.ipynb`
- **Figure 6**: The attention analysis is performed in two notebooks: `src/analysis/att_entropy.ipynb` and `src/analysis/att_interp.ipynb`.