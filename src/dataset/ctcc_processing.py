import anndata as ad
import numpy as np
import pandas as pd
import os

from datetime import datetime
from tqdm import tqdm  # Used for progress bars


if __name__ == "__main__":

    WORK_DIR = "/tudelft.net/staff-umbrella/ctcc/"
    DATA_FILE = "data/rosmap/combined_compressed.h5ad"
    DATA_PATH = os.path.join(WORK_DIR, DATA_FILE)
    
    OUT_FILE = "data/rosmap/ct_counts.h5ad"
    OUT_PATH = os.path.join(WORK_DIR, OUT_FILE)
    
    start_time = datetime.now()
    
    print(f"Loading dataset from file: {DATA_FILE}")
    adata = ad.read_h5ad(DATA_PATH)
    CELLTYPES = sorted(adata.obs["cell_type_high_resolution"].unique())
    DONORS = sorted(adata.obs["Donor ID"].unique())
    print("Done! Loaded AnnData object:")
    print(adata)
    print()
    
    # Initialize matrix
    n_celltypes = len(CELLTYPES)
    n_donors = len(DONORS)
    matrix = np.zeros((n_donors, n_celltypes))
    
    # Fill the matrix by counting the number of cells of each celltype for each donor
    for i, donor in enumerate(tqdm(DONORS, desc="")):
        adata_donor = adata[adata.obs["Donor ID"] == donor]
        donor_ct_counts = adata_donor.obs["cell_type_high_resolution"].value_counts()
        for j, celltype in enumerate(CELLTYPES):
            if celltype in donor_ct_counts:
                matrix[i, j] = donor_ct_counts[celltype]
    
    df = pd.DataFrame(matrix, index=DONORS, columns=CELLTYPES)
    print()
    
    # Aggregate the single-cell metadata per donor
    print("Transferring metadata...")
    obs = pd.DataFrame(index=DONORS)
    obs["n_cells"] = matrix.sum(axis=1)
    obs["n_celltypes"] = (matrix > 0).sum(axis=1)
    
    for obs_col in adata.obs.columns:
        if obs_col not in ["Subclass", "cell_type_high_resolution", "Celltype"]:
            obs[obs_col] = adata.obs.groupby("Donor ID")[obs_col].first()
            print(f" - {obs_col}")
    print("Done!\n")
    
    print("Combining to new AnnData object")
    adata_ct_counts = ad.AnnData(X=df, obs=obs)    
    print("Done! Resulting AnnData object:")
    print(adata_ct_counts)
    
    print(f"\nWriting to file: {OUT_FILE}")
    adata_ct_counts.write(OUT_PATH)
    print("Done!\n")
    
    print(f"Total time elapsed: {datetime.now() - start_time}\n")
    