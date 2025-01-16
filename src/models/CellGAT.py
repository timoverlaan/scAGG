import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import global_mean_pool, global_add_pool, global_max_pool, SAGPooling
from tqdm import tqdm, trange

from dataset.GraphDataset import GraphDataset
from models.aggregators import EasyAttentionAggregator, SelfAttentionAggregator, global_median_pool

class CellGAT(torch.nn.Module):
    name = "GAT"

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
    ) -> None:
        
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dropout = dropout
        self.heads = heads
        self.heads2 = heads2
        self.self_loops = self_loops
        self.task = task
        self.sex_covariate = sex_covariate

        super().__init__()

        self.gat1 = GATv2Conv(
            in_channels=dim_in,
            out_channels=dim_h,
            heads=heads,
            add_self_loops=self_loops,
        )

        self.gat2 = GATv2Conv(
            in_channels=dim_h * heads,
            out_channels=dim_h,
            heads=heads2,
            add_self_loops=self_loops,
        )

        if sag:
            self.SAG = SAGPooling(dim_h * heads2, ratio=SAGrate)
        else:
            self.SAG = None

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

    def forward(self, data: Data) -> torch.Tensor:        
        y_hat, _ = self.forward_with_embeddings(data)
        return y_hat

    def forward_with_embeddings(self, data: Data, ret_att: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = data.edge_index
        h = data.x
        msex = data.msex  # Added as a covariate

        # TODO: why use this dropout instead of pyg edge/node dropout?
        h = F.dropout(h, p=self.dropout, training=self.training)
        h, att_1 = self.gat1(h, edge_index, return_attention_weights=True)

        h = F.elu(h)
        h_1 = h

        h = F.dropout(h, p=self.dropout, training=self.training)
        h, att_2 = self.gat2(h, edge_index, return_attention_weights=True)

        h = F.elu(h)
        h_2 = h

        # Pooling
        if self.SAG is None:  # Skip SAG pooling
            h_graph = self.pooling(h, data.batch)
            msex_graph = global_mean_pool(msex, data.batch)
            h_pool = None
        else:
            pooled = self.SAG(h, edge_index, batch=data.batch)
            h, _, _, final_batch, _, _ = pooled
            h_pool = h
            h_graph = self.pooling(h, final_batch)
            msex_graph = global_mean_pool(msex, final_batch)

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

            if self.SAG is None:
                return y_pred_graph, embeddings, (att_1, att_2, None)

            # Reproduce SAG scores:
            sag_attn = h_2
            # sag_attn = sag_attn.unsqueeze(-1) if sag_attn.dim() == 1 else sag_attn
            att_pool = self.SAG.gnn(sag_attn, edge_index).view(-1)
            attention_scores = (att_1, att_2, att_pool)
        
            return y_pred_graph, embeddings, attention_scores
        else:
            return y_pred_graph, embeddings

    def attention_score(self, x_query: torch.Tensor, x_target: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        h_query = self.gat1.lin_l(x_query)
        h_target = self.gat1.lin_r(x_target)

        h = h_query + h_target
        h = F.leaky_relu(h, negative_slope=0.2)

        att = (h * self.gat1.att.flatten())
        att = att.reshape(att.shape[0], self.dim_h, self.heads).sum(dim=1)
        assert att.shape == (att.shape[0], self.heads)

        soft_att = F.softmax(att, dim=0)
        return att, soft_att

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
