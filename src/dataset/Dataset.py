import anndata as ad
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, adata: ad.AnnData, test: bool = False) -> None:
        self.adata = adata
        self.test = test  # Just here for consistency with the GraphDataset class

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        assert 0 <= index and index < self.adata.shape[0]

        feature = self.adata.X[index].toarray().flatten().astype('float32')
        label = self.adata.obs['y'][index]
        y = np.array([1 - label, label], dtype="float32")  # one-hot encoded

        return (
            torch.tensor(feature, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

    def __len__(self) -> int:
        return self.adata.shape[0]
