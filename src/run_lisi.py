import argparse
import os

import numpy as np
import anndata as ad

from analysis.external.lisi import compute_lisi


argparser = argparse.ArgumentParser(description='Compute Local Intrinsic Dimensionality (LISI) of a dataset')
argparser.add_argument('--data', type=str, help='Path to the dataset')
argparser.add_argument('--input', type=bool, help='Compute LISI of input space instead of embeddings', default=False)


if __name__ == '__main__':

    args = argparser.parse_args()
    data_path = args.data

    print(f"Path: {data_path}")
    assert os.path.exists(data_path), f"Path does not exist: {data_path}"

    file_name = data_path.split('/')[-1]

    # Load data
    print('Loading first data file...')
    adata = ad.read_h5ad(data_path)
    print('First data file loaded: ', adata)

    if args.input:
        print("\nComputing LISI scores of input space...")
        lisi_x = compute_lisi(np.array(adata.X.todense()), adata.obs[['Label']], label_colnames=["Label"], verbose=True, n_jobs=-1).flatten()
        
        # Write array to file
        np.savetxt(f"out/results/{file_name.replace('.h5ad', '_lisi_x.npy')}", lisi_x, delimiter=',')

        lisi_x = lisi_x.mean()
        print('LISI score of input space: ', lisi_x)
    
    else:
        print("\nComputing LISI scores of latent space... (dataset 1)")
        lisi_h1_f0 = compute_lisi(adata.obsm['h_2_f0'], adata.obs[['Label']], label_colnames=["Label"], verbose=True, n_jobs=-1).flatten()

        # Write array to file
        np.savetxt(f"out/results/{file_name.replace('.h5ad', '_lisi_h1_f0.npy')}", lisi_h1_f0, delimiter=',')

        lisi_h1_f0 = lisi_h1_f0.mean()
        print('LISI score of latent space (dataset 1): ', lisi_h1_f0)