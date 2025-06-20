import anndata as ad

import os


def split_celltype_adatas(adata: ad.AnnData, celltype_key: str) -> dict:
    
    """
    Splits an AnnData object into separate AnnData objects based on unique values in the specified cell type key.

    Parameters:
    adata (anndata.AnnData object): AnnData object to be split.
    celltype_key (str): Key in the `.obs` attribute of AnnData objects that contains cell type information.

    Returns:
    dict: A dictionary where keys are unique cell types and values are lists of AnnData objects corresponding to each cell type.
    """
    celltype_adatas = {}
    
    for celltype in adata.obs[celltype_key].unique():
        celltype_adatas[celltype] = adata[adata.obs[celltype_key] == celltype].copy()
    
    return celltype_adatas


if __name__ == "__main__":
    
    IN_PATH = "data/adata_rosmap_v3_top959.h5ad"  # TODO: Change to the top1000 genes on desktop later
    adata = ad.read_h5ad(IN_PATH)
    print(f"Loaded AnnData object with {adata.n_obs} observations and {adata.n_vars} variables.")
    
    adata.obs["Celltype"] = adata.obs["Supertype"].map({
        "Astrocytes": "Astrocytes",
        "Excitatory_neurons_set1": "Excitatory_neurons",
        "Excitatory_neurons_set2": "Excitatory_neurons",
        "Excitatory_neurons_set3": "Excitatory_neurons",
        "Immune_cells": "Immune_cells",
        "Inhibitory_neurons": "Inhibitory_neurons",
        "Oligodendrocytes": "Oligodendrocytes",
        "OPCs": "OPCs",
        "Vasculature_cells": "Vasculature_cells",
    })
    
    print("Splitting AnnData object by major cell types:")
    print(adata.obs["Celltype"].value_counts())
    major_celltype_adatas = split_celltype_adatas(adata, celltype_key="Celltype")
    
    print("Splitting AnnData object by minor cell types:")
    print(adata.obs["cell_type_high_resolution"].value_counts())
    minor_celltype_adatas = split_celltype_adatas(adata, celltype_key="cell_type_high_resolution")
    
    if not os.path.exists("data/rosmap959_split_major"):
        os.makedirs("data/rosmap959_split_major")  # TODO: Change to the top1000 genes on desktop later
    if not os.path.exists("data/rosmap959_split_minor"):
        os.makedirs("data/rosmap959_split_minor")
    
    # Save the split AnnData objects to file
    print("Saving major cell type AnnData objects...")
    for celltype, adata in major_celltype_adatas.items():
        adata.write_h5ad(f"data/rosmap959_split_major/{celltype.replace(' ', '_').replace('\\', '-').replace('/', '-')}.h5ad")
            
    print("Saving minor cell type AnnData objects...")
    for celltype, adata in minor_celltype_adatas.items():
        adata.write_h5ad(f"data/rosmap959_split_minor/{celltype.replace(' ', '_').replace('\\', '-').replace('/', '-')}.h5ad")
            
    print("Done.")