import os
import json

import anndata as ad
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import NeighborLoader as PygNeighborLoader
from tqdm import trange

from dataset.GraphDataset import GraphDataset
from dataset.split import adata_kfold_split
from models.NoGraph import NoGraph
from train_util import _train_epoch, generate_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_device = "cpu"


def calc_fold_performance(adata: ad.AnnData, split_i: int, donors: list) -> dict:
    """
    Calculate the performance of the model on the test set of the fold.
    """
    y_true = np.array([adata[adata.obs["Donor ID"] == donor_id].obs["y"].mean() for donor_id in donors], dtype=int)    
    y_pred = np.concatenate([adata.uns[f"y_pred_graph_f{split_i}"][str(donor_id)] for donor_id in donors])
    y_pred_hard = np.argmax(y_pred, axis=1)

    return {
        "accuracy": accuracy_score(y_true, y_pred_hard),
        "precision": precision_score(y_true, y_pred_hard),
        "recall": recall_score(y_true, y_pred_hard),
        "f1": f1_score(y_true, y_pred_hard),
        "roc_auc": roc_auc_score(y_true, y_pred[:, 1]),
    }

def calc_var_batched(adata: ad.AnnData, chunk_size: int = 10000) -> tuple:
    """
    Calculate the mean and standard deviation of the data in an AnnData object in batches.
    
    Parameters:
    adata (anndata.AnnData): The AnnData object containing the data.
    chunk_size (int): The size of the chunks to process at a time.
    
    Returns:
    tuple: A tuple containing the means and standard deviations as torch tensors.
    """
    sums = np.zeros((1, adata.shape[1]))
    sum_sqs = np.zeros((1, adata.shape[1]))
    
    for i in trange(0, adata.shape[0], chunk_size):
        chunk = adata.X[i:i+chunk_size].todense()
        sums += chunk.sum(axis=0)
        sum_sqs += (np.power(chunk, 2)).sum(axis=0)

    means = sums / adata.shape[0]
    stds = np.sqrt(sum_sqs / adata.shape[0] - np.power(means, 2))
    stds[stds == 0] = 1  # Avoid division by zero
    return torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32)


def run(adata: ad.AnnData, hp: dict, n_splits: int) -> dict:
    """
    Runs cross-validated training and evaluation of a classification model on single-cell AnnData.
    
    Args:
        adata (ad.AnnData): Annotated data matrix containing single-cell data with donor and pathology information.
        hp (dict): Dictionary of hyperparameters for model configuration and training.
        n_splits (int): Number of cross-validation splits (folds) to use.
        split_seed (int): Random seed for reproducible data splitting.
        n_epochs (int): Number of training epochs per fold.
        
    Returns:
        dict: Dictionary containing performance metrics and results for each cross-validation fold.
    """
    
    donor_counts = adata.obs["Donor ID"].value_counts()
    
    # Remap labels based on the @Wang2022 approach
    df = adata.obs.groupby("Donor ID").first()
    wang_labels = dict()
    for i in df.index:
        if df.loc[i]["cogdx"] == 4 and df.loc[i]["braaksc"] >= 4 and df.loc[i]["ceradsc"] <= 2:
            wang_labels[i] = "AD"
        elif df.loc[i]["cogdx"] == 1 and df.loc[i]["braaksc"] <= 3 and df.loc[i]["ceradsc"] >= 3:
            wang_labels[i] = "CT"
        else:
            wang_labels[i] = "Other"
    cell_labels = np.empty(adata.n_obs, dtype=object)
    for donor, label in wang_labels.items():
        cell_labels[adata.obs["Donor ID"] == donor] = label
    adata.obs["Label"] = cell_labels

    donor_labels = adata.obs.groupby("Donor ID").first()["Label"]
    print("Donor labels:", adata.obs["Label"].value_counts())
    if "AD" in adata.obs["Label"].unique() and "CT" in adata.obs["Label"].unique():
        adata = adata[adata.obs["Label"].isin(["AD", "CT"])].copy()
        adata.obs["y"] = adata.obs["Label"].map({"AD": 1, "CT": 0})

    test_perfs = dict()

    for split_i, (train_donors, test_donors) in enumerate( \
        adata_kfold_split(adata, n_splits=n_splits, seed=hp["split_seed"], task="classification")
    ):
        # Split the AnnData object based on the donors
        train_adata = adata[adata.obs["Donor ID"].isin(train_donors), :].copy()
        test_adata = adata[adata.obs["Donor ID"].isin(test_donors), :].copy()

        means, stds = calc_var_batched(train_adata)
    
        train_bal = train_adata.obs["y"].value_counts()[1] / train_adata.n_obs
        test_bal = test_adata.obs["y"].value_counts()[1] / test_adata.n_obs
        print(f"Train balance p(AD)={train_bal:.3f}")
        print(f"Test balance p(AD)={test_bal:.3f}")

        # Only stratify on class label
        class_weight = {
            adata.obs["Label"].unique()[0]: 0,
            adata.obs["Label"].unique()[1]: 0,
        }
        for donor_id in train_donors:
            class_weight[donor_labels[donor_id]] += 1

        # We sample according to class prior and correct for graph size,
        #   so large graphs are not sampled from more frequently.
        donor_weights = dict()
        for donor_id in train_donors:
            donor_weights[donor_id] = 1000 / donor_counts[donor_id]
        sample_weights = torch.tensor([donor_weights[donor_id] for donor_id in train_adata.obs["Donor ID"]], dtype=torch.float32)

        # The train data is loaded in batches from different donors, to provide regularization
        train_loader = PygNeighborLoader(
            data=GraphDataset(adata=train_adata, test=False)[0],
            num_neighbors=[15, 15],
            batch_size=1,  # We manually build the batches in the training loop
            shuffle=False,
            subgraph_type="induced",
            sampler=WeightedRandomSampler(weights=sample_weights, num_samples=train_adata.n_obs, replacement=True),
        )

        model = NoGraph(
            dim_in=hp["dim_in"],
            dim_h=hp["dim_h"],
            dropout=hp["dropout"],
            heads=hp["heads"],
            heads2=hp["heads2"],
            self_loops=hp["self_loops"],
            SAGrate=0.5,
            pooling="mean",
            sag=False,
            task="classification",
            sex_covariate=True,
            means = means,
            stds = stds,
        )

        model = model.to(device)
        model.train()

        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=hp["lr"],
            weight_decay=hp["wd"],
        )

        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
            
        # Train the model
        for epoch in range(hp["n_epochs"]):
            _ = _train_epoch(model, train_loader, optimizer, criterion, device, epoch, \
                n_batches=min(train_adata.n_obs // hp["batch_size"] + 1, 1000), hp=hp)
        del train_adata, train_loader

        model = model.to(test_device)
        model.eval()

        test_adata = generate_embeddings(test_adata, model, test_device, hp, {}, split_i, test_donors, train_donors, save_attention=False)
        test_perfs[split_i] = calc_fold_performance(test_adata, split_i, test_donors)
        
        print(f"Fold {split_i} performance: {test_perfs[split_i]}\n\n")
            
    return test_perfs


if __name__ == "__main__":
    
    hp={
        "model_type": "NoGraph",
        "lr": 0.001,
        "wd": 0.00005,
        "batch_size": 8,
        "dropout": 0.1,
        "dim_in": 959,
        "dim_h": 32,
        "heads": 8,
        "heads2": 4,
        "self_loops": False,
        "n_epochs": 2,
        "split_seed": 42,
        "pooling": "mean",
    }
    N_SPLITS = 5
    
    major_files = os.listdir("data/rosmap959_split_major")  # TODO: change to top1000 files on desktop later
    minor_files = os.listdir("data/rosmap959_split_minor")
    
    major_results = dict()
    minor_results = dict()
    
    for file in major_files:
        adata = ad.read_h5ad(f"data/rosmap959_split_major/{file}")  # TODO: change to top1000 files on desktop later
        ct_name = adata.obs["Celltype"].unique()[0]
        print(f"Major cell type {ct_name} has {adata.n_obs} observations and {adata.n_vars} variables.")
        
        results = run(adata, hp, n_splits=N_SPLITS)
        print(f"Results for {ct_name}: {results}")
        major_results[ct_name] = results
        major_results[ct_name]["abundances"] = adata.obs["Donor ID"].value_counts().to_dict()
        
        with open("out/results/rosmap959_major.json", "w") as f:  # TODO: change to top1000 files on desktop later
            json.dump(major_results, f, indent=4)
        
    for file in minor_files:
        adata = ad.read_h5ad(f"data/rosmap959_split_minor/{file}")  # TODO: change to top1000 files on desktop later
        ct_name = adata.obs["cell_type_high_resolution"].unique()[0]
        print(f"Minor cell type {ct_name} has {adata.n_obs} observations and {adata.n_vars} variables.")
        
        results = run(adata, hp, n_splits=N_SPLITS)
        print(f"Results for {ct_name}: {results}")
        minor_results[ct_name] = results
        minor_results[ct_name]["abundances"] = adata.obs["Donor ID"].value_counts().to_dict()

        with open("out/results/rosmap959_minor.json", "w") as f:  # TODO: change to top1000 files on desktop later
            json.dump(minor_results, f, indent=4)
            