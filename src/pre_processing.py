"""
VERSION = v3

This file contains the preprocessing pipeline for the data, including the following steps:
- Load the cell expression data
- Perform standard scanpy QC filtering
- Select highly variable genes (seurat v1 or v3)
- Calculate PCA
- Build KNN graphs per donor

The output of this script is a set of AnnData files, that can be used for training the GNNs.

This script requires ~500GB of memory to run, and takes ~1 hour to complete.
"""

import os
from datetime import datetime
import argparse

import anndata as ad
import numpy as np
import scanpy
from tqdm import tqdm, trange
import scanpy as sc
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Data pre-processing pipeline')
parser.add_argument("--input", type=str, 
                    help="Path to input .h5ad file")
parser.add_argument('--output', type=str,
                    help='Path to output .h5ad file')
parser.add_argument('--n_top_genes', type=int, default=1000,
                    help='Number of top genes to select')
parser.add_argument('--gene_selection', type=str, default="seurat_v1",
                    help='Gene selection method to use. Options: [seurat_v1, seurat_v3]')
parser.add_argument('--k_neighbors', type=int, default=30,
                    help='Number of neighbors to use for KNN graph construction')
parser.add_argument("--regress", action="store_true", default=False,
                    help="Regress out total counts and mitochondrial counts")
parser.add_argument("--scale", action="store_true", default=False,
                    help="Scale data to unit variance and zero mean")


def build_knn_graphs(adata: ad.AnnData, k_neighbors: int) -> ad.AnnData:
    """
    Use scanpy to make a KNN graph per donor.
    The AnnData object is first sliced per donor, then the KNN graph is calculated for each slice.
    Finally, the slices are concatenated into a single AnnData object.

    Args:
        adata (ad.AnnData)
        k_neighbors (int)

    Returns:
        ad.AnnData
    """

    slices = []
    for donor in tqdm(adata.obs["Donor ID"].unique(), desc="Building donor KNN graphs"):
        donor_slice = adata[adata.obs["Donor ID"] == donor]
        scanpy.pp.neighbors(
            donor_slice, n_neighbors=k_neighbors, use_rep="X_pca")
        slices.append(donor_slice)

    return ad.concat(
        slices,
        merge="same",
        uns_merge="same",
        label="all",
        index_unique="-",
        pairwise=True,
    )


if __name__ == "__main__":

    start = datetime.now()
    print("Starting pre-processing pipeline...")

    args = parser.parse_args()
    assert args.gene_selection in {"seurat_v1", "seurat_v3"}
    assert os.path.exists(args.input), f"File not found: {args.input}"
    print("Arguments:", args)
        
    # Then cell expression data (requires ????GB memory)
    print("Loading raw AnnData to memory...")
    adata = ad.read_h5ad(filename=args.input)
    print(f"AnnData loaded, shape={adata.shape}\n")

    # Now we do all of the scanpy processing, following this tutorial:
    # https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html#preprocessing

    print("Filtering cells (with <200 genes) and genes (detected in <200 cells)...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=200)
    print(f"Filtered data, shape={adata.shape}\n")

    print("Calculating QC metrics, and saving plots...")
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    print("MT genes detected:", adata.var["mt"].sum())
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        stripplot=False,
    )
    plt.savefig("out/figures/processing_qc_metrics.png")
    sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt")
    plt.savefig("out/figures/processing_mt_counts.png")
    sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")
    plt.savefig("out/figures/processing_n_genes.png")
    print("Done with QC metrics\n")

    print("Dropping cells with >5% mitochondrial counts...")
    adata = adata[adata.obs.pct_counts_mt < 5, :].copy()
    print(f"Filtered data, shape={adata.shape}\n")

    # (As seurat_v3 requires raw count data, we do this before log-normalization)
    if args.gene_selection == "seurat_v3":
        print("Selecting highly variable genes using Seurat v3 method... (before log-normalization)")
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=args.n_top_genes)
        sc.pl.highly_variable_genes(adata)
        plt.savefig(f"out/figures/{args.slurm_job}_hvgs.png")
        adata_hvg = adata[:, adata.var.highly_variable].copy()
        del adata
        adata = adata_hvg
        print(f"Selected {adata.n_vars} highly variable genes")

    print("Normalizing total counts and log-transforming...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("Done with normalization and log-transforming\n")

    if args.gene_selection == "seurat_v1":
        print("Selecting highly variable genes...")
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=args.n_top_genes)
        sc.pl.highly_variable_genes(adata)
        plt.savefig(f"out/figures/{args.slurm_job}_hvgs.png")
        adata_hvg = adata[:, adata.var.highly_variable].copy()
        del adata
        adata = adata_hvg
        print(f"Selected {adata.n_vars} highly variable genes")

    if args.regress:
        print("Regressing out total counts and mitochondrial counts...")
        sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])

    if args.scale:
        print("Scaling data to unit variance")
        import scipy.sparse as sp
        # sc.pp.scale(adata, max_value=10, zero_center=False)
        
        X = adata.X.copy()
        del adata.X
        
        sigmas = np.zeros(X.shape[1])
        for gene in trange(X.shape[1], desc="Calculating gene nz-sigmas"):
            x_col = X[:, gene]
            x_col_nz = x_col[x_col != 0]
            sigmas[gene] = np.std(x_col_nz)            
            
        adata.X = sp.csr_matrix(X / np.array(sigmas))
        del X

    # We add a column 'y', that is 1.0 for AD, and 0.0 for CT
    adata.obs["y"] = np.where(adata.obs["Label"] == "AD", 1.0, 0.0)

    # First we calculate PCA on the full dataset
    print("Caclulating PCA...")
    scanpy.pp.pca(adata, n_comps=50)
    assert "X_pca" in adata.obsm  # scanpy's PCA should have added this
    print("Finished PCA\n")

    # Now we can build the KNN graphs
    print(f"Building KNN graphs... (k={args.k_neighbors})")
    adata_knn = build_knn_graphs(adata, k_neighbors=args.k_neighbors)
    del adata
    adata = adata_knn
    print("Done building KNN graphs\n")

    # And save the resulting file
    print(f"Saving result to {args.output}...")
    adata.write_h5ad(filename=args.output, compression="gzip")
    print("Done!\n")
    
    end = datetime.now()
    print(f"Took {end-start} to complete.")
