import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

# --- CONFIGURATION ---
H5AD_FILE = "out/results/GAT_2024-05-23-16-24-07_results.h5ad"
PCA_KEY = "h_2_f0"  # Change if needed
N_PCS = 5           # Number of PCs to compute

# --- HELPER FUNCTION ---
def compute_gene_pca1_corr(adata, pca_key, n_pcs=5):
    """
    Computes Spearman correlation between each gene's expression and the first PCA component.
    Returns a pandas Series with gene names as index and correlation values as data.
    """
    # Run PCA on the specified embedding
    pca = PCA(n_components=n_pcs)
    x_pca = pca.fit_transform(adata.obsm[pca_key].toarray())
    spearman_corrs = []
    for i, gene in enumerate(tqdm(adata.var_names, desc="Genes")):
        gene_expr = adata.X[:, i].toarray().flatten()
        if np.var(gene_expr) == 0:
            spearman_corrs.append(np.nan)
            continue
        corr, _ = spearmanr(gene_expr, x_pca[:, 0])
        spearman_corrs.append(corr)
    return pd.Series(spearman_corrs, index=adata.var_names)

if __name__ == "__main__":
        
    # --- LOAD DATA ---
    adata_full = ad.read_h5ad(H5AD_FILE)
    
    # combine all three excitatory neuron sets into one
    # excitatory_mask = adata_full.obs["Supertype"].isin(["Excitatory_neurons_set1", "Excitatory_neurons_set2", "Excitatory_neurons_set3"])
    # adata_full.obs.loc[excitatory_mask, "Supertype"] = "Excitatory_neurons"
    adata_full.obs["Supertype"].rename_categories(
        {"Excitatory_neurons_set1": "Excitatory_neurons",
         "Excitatory_neurons_set2": "Excitatory_neurons",
         "Excitatory_neurons_set3": "Excitatory_neurons"}, inplace=True)
   
   
    # --- MAIN ANALYSIS ---

    # Get all unique major and minor cell types
    major_types = adata_full.obs["Supertype"].unique()
    minor_types = adata_full.obs["cell_type_high_resolution"].unique()

    # Prepare result DataFrames
    major_corrs = pd.DataFrame(index=adata_full.var_names)
    minor_corrs = pd.DataFrame(index=adata_full.var_names)

    # Compute for each major cell type
    for celltype in tqdm(major_types, desc="Major cell types"):
        adata = adata_full[adata_full.obs["Supertype"] == celltype]
        if adata.n_obs < 10:  # Skip very small groups
            continue
        corrs = compute_gene_pca1_corr(adata, PCA_KEY, N_PCS)
        major_corrs[celltype] = corrs

    # Compute for each minor cell type
    for celltype in tqdm(minor_types, desc="Minor cell types"):
        adata = adata_full[adata_full.obs["cell_type_high_resolution"] == celltype]
        if adata.n_obs < 10:  # Skip very small groups
            continue
        corrs = compute_gene_pca1_corr(adata, PCA_KEY, N_PCS)
        minor_corrs[celltype] = corrs

    # --- SAVE RESULTS ---
    major_corrs.to_csv("out/results/gene_pca1_corrs_major.csv")
    minor_corrs.to_csv("out/results/gene_pca1_corrs_minor.csv")

    # --- OPTIONAL: Print summary ---
    print("Major cell type correlation matrix shape:", major_corrs.shape)
    print("Minor cell type correlation matrix shape:", minor_corrs.shape)