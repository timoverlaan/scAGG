import argparse

import anndata as ad
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Generate UMAP embeddings a set of embeddings from train.py')
parser.add_argument('--input', type=str, help='Path to the results data (.h5ad)')
parser.add_argument('--output', type=str, default=None, help='Path to save the UMAP embeddings (.h5ad)')
parser.add_argument('--pcs', type=int, default=50, help='Number of PCs to use for UMAP')
parser.add_argument('--n-neighbors', type=int, default=15, help='Number of neighbors for UMAP')
parser.add_argument('--min-dist', type=float, default=0.1, help='Minimum distance for UMAP')
parser.add_argument('--downsample', type=int, default=None, help='Downsample the data to this size')
parser.add_argument('--graph-level', action="store_true", help="Use graph-level embeddings if True")
parser.add_argument('--split-heads', action="store_true", help="Use multi-head attention for separate UMAPs")
parser.add_argument('--layer', type=int, default=2, help="Layer to use (1=after 1st, 2=after 2nd)")
parser.add_argument('--draw', action="store_true", help="Draw the UMAPs using matplotlib?")
parser.add_argument('--save', action="store_true", help="Save the UMAPs to AnnData file?")
parser.add_argument('--save-fig', action="store_true", help="Save the UMAPs to PNG file?")
parser.add_argument('--fold', type=int, default=0, help="Fold from cross-validation to use (0-4)")


if __name__ == "__main__":


    ############################################### TODO: THIS IS VERY MUCH A WORK IN PROGRESS !!

    args = parser.parse_args()

    # Load the data
    adata = ad.read_h5ad(args.input)

    if args.graph_level:

        # TODO: this is a work in progress
        # I'll first play with this in a notebook to get it working, then move it here

        # Load the graph-level embeddings
        h_graph = adata.uns["h_graph"]
        donors, h_graph = [key for key in h_graph.keys()], [h_graph[key] for key in h_graph.keys()]
        h_graph = np.concatenate(h_graph, axis=0)
        print(f"h_graph.shape={h_graph.shape}")


    if args.downsample is not None:
        # Downsample the data
        print(f"Downsampling to {args.downsample} cells")
        all_idx = list(range(adata.n_obs))
        chosen = np.random.choice(all_idx, size=args.downsample, replace=False)
        adata = adata[chosen, :].copy()
        print(f"Downsample complete. adata.shape={adata.shape}")

    ##############################
    #  UMAP for input space
    ##############################

    # If there's already a UMAP, we delete it, because we want to recompute it for consistency
    if "X_umap" in adata.obsm.keys():
        del adata.obsm["X_umap"]

    print("Calculating PCA for input space")
    adata.obsm["X_pca"] = PCA(
        n_components=args.pcs).fit_transform(np.asarray(adata.X.todense()))

    print("Calculating UMAP for input space")
    # Compute the UMAP embedding, TODO: look into rapids.ai for faster UMAP
    adata.obsm["X_umap"] = UMAP(
        verbose=True,
        low_memory=False,
        n_neighbors=args.n_neighbors,
    ).fit_transform(adata.obsm["X_pca"])

    if args.draw:
        plt.scatter(adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1], s=0.1)
        plt.show()

    # Now we also make a UMAP embedding of the latent space stored in obsm["X_embedding"]
    # But first we have to calculate PCA for it
    print("Calculating PCA for latent space")
    if args.graph_level:
        latent_space = "h_graph"
        raise NotImplementedError("TODO: graph-level embeddings")
    else:
        latent_space = f"h_1_f{args.fold}" if args.layer == 1 else f"h_2_f{args.fold}"

    adata.obsm["h_pca"] = PCA(
        n_components=args.pcs).fit_transform(adata.obsm[latent_space])

    print("Calculating UMAP for latent space")
    adata.obsm["h_umap"] = UMAP(
        verbose=True,
        low_memory=False,
        n_neighbors=args.n_neighbors,
    ).fit_transform(adata.obsm["h_pca"])

    if args.draw:
        plt.scatter(adata.obsm["h_umap"][:, 0], adata.obsm["h_umap"][:, 1], s=0.1)
        plt.show()

    if args.save:
        print("Saving UMAP embeddings to file")
        if args.output is not None:
            outfile = args.output
        else:
            outfile = args.input.replace(".h5ad", "_umap.h5ad")
        adata.write_h5ad(outfile)
        print(f"UMAP embeddings saved to {outfile}")
    
    print("\nDone.\n")
