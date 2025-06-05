import anndata as ad
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.data import Data
from torch_geometric.nn.pool import global_mean_pool, global_add_pool, global_max_pool
from tqdm import tqdm, trange
from models.aggregators import EasyAttentionAggregator, SelfAttentionAggregator, global_median_pool
from dataset.GraphDataset import GraphDataset


means = None
stds = None


class NoGraph(torch.nn.Module):
    name = "NoGraph"

    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        dropout: float,
        heads: int,
        heads2: int,
        SAGrate: float,
        self_loops: bool,
        pooling: str = "mean",
        sag: bool = True,
        task: str = "classification",
        sex_covariate: bool = False,
        means: torch.Tensor = None,
        stds: torch.Tensor = None,
    ) -> None:

        super().__init__()

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dropout = dropout
        self.heads = heads
        self.heads2 = heads2
        self.self_loops = self_loops
        self.task = task
        self.sex_covariate = sex_covariate

        self.layer1 = torch.nn.Linear(dim_in, dim_h * heads)

        self.layer2 = torch.nn.Linear(dim_h * heads, dim_h * heads2)

        ATT_POOL_HEADS = 4

        if pooling == "mean":
            self.pooling = global_mean_pool
        elif pooling == "max":
            self.pooling = global_max_pool
        elif pooling == "add":
            self.pooling = global_add_pool
        elif pooling == "median":
            self.pooling = global_median_pool
        elif pooling == "self-att-sum":
            self.pooling = SelfAttentionAggregator(dim_h * heads2, heads=ATT_POOL_HEADS, cat=False)
        elif pooling == "self-att-cat":
            self.pooling = SelfAttentionAggregator(dim_h * heads2, heads=ATT_POOL_HEADS, cat=True)
        elif pooling == "basic-att-sum":
            self.pooling = EasyAttentionAggregator(dim_h * heads2, ATT_POOL_HEADS, cat=False)
        elif pooling == "basic-att-cat":
            self.pooling = EasyAttentionAggregator(dim_h * heads2, ATT_POOL_HEADS, cat=True)

        classifier_dim = dim_h * heads2
        if "att" in pooling and self.pooling.cat:
            classifier_dim *= self.pooling.num_heads    
        if sex_covariate:
            classifier_dim += 1

        self.classifier = torch.nn.Linear(
            in_features=classifier_dim,
            out_features=2 if task == "classification" else 1,
        )

        # For normalization
        self.means = torch.nn.Parameter(means, requires_grad=False) if means is not None else None
        self.stds = torch.nn.Parameter(stds, requires_grad=False) if stds is not None else None


    def forward(self, data: Data) -> torch.Tensor:        
        y_hat, _ = self.forward_with_embeddings(data)
        return y_hat

    def forward_with_embeddings(self, data: Data, ret_att: bool = False) -> tuple[torch.Tensor, torch.Tensor]:        
        edge_index = data.edge_index
        h = data.x
        msex = data.msex  # Added as a covariate

        # Normalize the input features
        if self.means is not None and self.stds is not None:
            h = h - self.means
            h = h / self.stds

        # TODO: why use this dropout instead of pyg edge/node dropout?
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer1(h)

        h = F.elu(h)
        h_1 = h

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer2(h)

        h = F.elu(h)
        h_2 = h

        # Pooling
        if self.pooling == global_mean_pool:
            h_graph = global_mean_pool(h, data.batch)
            att_pool = None
        else:
            h_graph, att_pool = self.pooling(h, data.batch, return_att=True)

        msex_graph = global_mean_pool(msex, data.batch)
        h_pool = None

        h_graph_save = h_graph  # Saved embeddings should not include the sex covariate
        if self.sex_covariate:
            h_graph = torch.cat([h_graph, msex_graph.view(-1, 1)], dim=-1)

        if self.task == "regression":
            y_pred_graph = self.classifier(h_graph).view(-1)  # TODO: check if view is necessary
        else:
            assert self.task == "classification"
            y_pred_graph = F.softmax(self.classifier(h_graph), dim=-1)
        
        embeddings = (h_1, h_2, h_pool, h_graph_save)

        if ret_att:
            return y_pred_graph, embeddings, (None, None, att_pool)
        else:
            return y_pred_graph, embeddings

    def _embed(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Calculates the embedding for a single donor (in-place)
        """
        donor_graph = GraphDataset(adata=adata, test=False)[0]

        # Forward pass and get the embeddings
        y_pred, embeddings = self.forward_with_embeddings(data=donor_graph)

        # Store in the sliced AnnData object
        adata.obsm["X_embedding"] = embeddings.cpu().detach().numpy()
        adata.obsm["y_pred"] = y_pred.cpu().detach().numpy()

        return adata

    def calc_embeddings(self, adata: ad.AnnData) -> ad.AnnData:

        if len(adata.obs["Donor ID"].unique()) == 1:
            return self._embed(adata)

        adata_idxs_all = []
        embeddings_all = []
        y_pred_all = []
        
        for donor in tqdm(adata.obs["Donor ID"].unique(), desc="Embedding per donor"):
            # Slice the donor from the full data and contruct a graph from it
            donor_slice = adata[adata.obs["Donor ID"] == donor]
            donor_graph = GraphDataset(adata=donor_slice, test=False)[0]

            # Forward pass and get the embeddings
            y_pred, embeddings = self.forward_with_embeddings(data=donor_graph)
    
            embeddings = embeddings.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
            adata_idxs = donor_slice.obs_names.to_numpy()
            adata_idxs_all.append(adata_idxs)
            embeddings_all.append(embeddings)
            y_pred_all.append(y_pred)

        # Recombine embeddings and predictions in correct order
        adata_idxs_all = np.concatenate(adata_idxs_all)
        adata.obsm["X_embedding"] = np.concatenate(embeddings_all)
        adata.obsm["y_pred"] = np.concatenate(y_pred_all)

        # Sanity check
        for i in range(len(adata_idxs_all)):
            assert adata.obs_names[i] == adata_idxs_all[i], "Embeddings order is wrong!"

        return adata
