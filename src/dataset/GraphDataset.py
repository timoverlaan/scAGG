from abc import ABC

import anndata as ad
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from dataset.split import DONOR_COL


def _adata_to_pyg_data(adata: ad.AnnData) -> Data:
    # Get the edges from the current donor
    edges_in, edges_out = adata.obsp['connectivities'].nonzero()
    edge_list = np.array([edges_in, edges_out])

    # One-hot encode the labels
    y = np.array([1 - adata.obs['y'], adata.obs['y']])
    msex = adata.obs['msex'].to_numpy()

    return Data(
        x=torch.tensor(adata.X.toarray(), dtype=torch.float32),
        edge_index=torch.tensor(edge_list, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.float32).T,
        msex=torch.tensor(msex, dtype=torch.long)
    )


class GraphDataset(InMemoryDataset, ABC):

    def __init__(self, adata: ad.AnnData, test: bool = False) -> None:
        super().__init__()

        self.adata = adata
        self.n_cells = adata.n_obs  # number of cells
        self.n_genes = adata.n_vars  # number of genes

        self.graphs = []
        self.metadata = dict()  # Used to store covariates

        if test:  # If test mode is set, separate the donor graphs from each other
            for donor in adata.obs[DONOR_COL].unique():
                # Slice data of current donor, convert to pyg data and append to data list
                self.graphs.append(
                    _adata_to_pyg_data(adata[adata.obs[DONOR_COL] == donor])
                )
        else:
            # If test mode is not set, combine all donor graphs into 1 graph
            self.graphs.append(
                _adata_to_pyg_data(adata)
            )

    def len(self) -> int:
        return len(self.graphs)

    def get(self, i: int) -> Data:
        return self.graphs[i]
