from datetime import datetime
from typing import Union

import gc
import psutil
import anndata as ad
import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data, Batch, InMemoryDataset
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.loader import NeighborLoader as PygNeighborLoader
from torch_geometric.nn.pool import global_mean_pool
from tqdm import tqdm

from dataset.Dataset import Dataset
from dataset.GraphDataset import GraphDataset
from dataset.split import adata_train_test_split
from models.CellGAT import CellGAT
from models.Linear import Linear
from models.NoGraph import NoGraph

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


CACHE_CLEAN_FREQ = 100  # Clean the cache every n batches


def mem(tag: str = ""):
    mem_info = psutil.Process().memory_info()
    try:
        mem_ps, mem_peak = mem_info.rss, mem_info.peak_wset
        print(f"MEMORY USAGE (current={mem_ps / 10**6:>6.0f} MB, peak={mem_peak / 10**6:>6.0f} MB): {tag}")
    except:
        # TODO: peak_wset is only available on windows
        mem_ps = mem_info.rss
        print(f"MEMORY USAGE (current={mem_ps / 10**6:>6.0f} MB): {tag}")


def _prepare_data_for_model(data, model: Union[CellGAT, Linear], device) -> tuple[Union[Data, torch.Tensor], torch.Tensor]:
    """
    Prepare data for the specific model.
    The GAT and baseline Linear model require different data preparation.
    This function allows using the same train and test functions for both models.

    Args:
        data (tuple): Tuple of data and labels.
        model (Union[GAT, Linear]): Model to use.
        device (torch.device): Device to use.

    Returns:
        tuple[Any, torch.Tensor]: Prepared data and labels.
    """
    if type(model) == Linear:
        x, y_true = data
        data = x.to(device)
        y_true = y_true.to(device)
    elif type(model) == CellGAT or type(model) == NoGraph:
        data = data.to(device)
        y_true = data.y
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    return data, y_true


class BatchSet(InMemoryDataset):

    def __init__(self, data_list: list):
        super().__init__()        
        self.data_list = data_list

    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        print("get:", idx)
        return self.data_list[idx]



def _train_epoch(model, loader, optimizer, criterion, device, epoch, n_batches, hp: dict) -> tuple[float, float]:

    model.train()

    train_loss = 0
    train_acc = 0
    train_mse = 0
    train_r2 = 0
    train_mae = 0
    train_samples = 0  # Used to compute the average loss

    y_preds = []
    y_trues = []

    with tqdm(
        loader,
        desc=f"Training epoch={epoch}",
        total=n_batches * hp["batch_size"],
        colour="blue",
    ) as pbar:
        
        next_batch = []
        for data in pbar:

            if len(next_batch) < hp["batch_size"]:
                del data.batch_size
                next_batch.append(data)
                continue

            data = Batch.from_data_list(next_batch)
            next_batch = []

            x, y_true = _prepare_data_for_model(
                data, model, device)
            
            # if model.pool is not None or model.pool_str == "att":
            # We make sure the labels are aggregated per batch
            y_true = global_mean_pool(y_true, data.batch)
            if model.task == "regression":
                y_true = y_true[:, 1]  # (then the first column doesn't make any sense)

            y_trues.append(y_true.cpu().detach().numpy())

            optimizer.zero_grad()
            y_pred = model(data)

            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            y_preds.append(y_pred.cpu().detach().numpy())

            # Save loss and accuracy
            train_loss += loss.item()
            if model.task == "classification":
                train_acc += float(sum(y_pred.argmax(dim=1) == y_true.argmax(dim=1)))
            train_samples += len(y_true)

            # Set preliminary average loss in the progress bar
            yt_cat = np.concatenate(y_trues)
            yp_cat = np.concatenate(y_preds)
            if model.task == "classification":
                yt_cat_argmax = yt_cat.argmax(axis=1)
                yp_cat_argmax = yp_cat.argmax(axis=1)
            elif model.task == "regression":
                yt_cat_argmax = yt_cat
                yp_cat_argmax = yp_cat

            if len(y_preds) > 1:
                if model.task == "classification":
                    try:
                        pbar.set_postfix(
                            loss=f"{train_loss/train_samples:.3f}",
                            acc=f"{accuracy_score(yt_cat_argmax, yp_cat_argmax):.3f}",
                            f1=f"{f1_score(yt_cat_argmax, yp_cat_argmax):.3f}",
                            prec=f"{precision_score(yt_cat_argmax, yp_cat_argmax, zero_division=0):.3f}",
                            rec=f"{recall_score(yt_cat_argmax, yp_cat_argmax):.3f}",
                            auc=f"{roc_auc_score(yt_cat, yp_cat):.3f}",
                            bias=f"{(yp_cat_argmax == 1).mean():.3f}",  # How many are predicted as AD
                        )
                    except:
                        pass
                        # print("yt_cat_argmax", yt_cat_argmax)
                        # print("yp_cat_argmax", yp_cat_argmax)
                elif model.task == "regression":
                    pbar.set_postfix(
                        loss=f"{train_loss/train_samples:.3f}",
                        mse=f"{mean_squared_error(yt_cat, yp_cat):.3f}",
                        r2=f"{r2_score(yt_cat, yp_cat):.3f}",
                        mae=f"{mean_absolute_error(yt_cat, yp_cat):.3f}",
                    )

            if pbar.n % CACHE_CLEAN_FREQ == 0:
                gc.collect()
                torch.cuda.empty_cache()

            pbar.update(hp["batch_size"])

            # Stop training after n_batches
            if pbar.n + hp["batch_size"] >= pbar.total:
                pbar.update(pbar.total - pbar.n)
                pbar.refresh()
                break

    if model.task == "regression":

        # calculate the correlation between the true and predicted values
        r = np.corrcoef(yt_cat, yp_cat)[0, 1]

        # import and use spearman correlation
        from scipy.stats import spearmanr
        r_spearman = spearmanr(yt_cat, yp_cat)[0]

        plt.figure()
        plt.scatter(yt_cat, yp_cat)
        plt.title(f"True vs predicted, epoch={epoch} (train set), r={r:.3f}, r_sp={r_spearman:.3f}")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.show(block=True)

    return train_loss / train_samples, train_acc / train_samples


def _test_epoch(model, loader, criterion, device, epoch) -> tuple[float, float]:

    test_loss = 0
    test_acc = 0
    test_samples = 0

    model.eval()

    y_preds = []
    y_trues = []

    with tqdm(loader, desc=f"Testing epoch={epoch}", colour="green") as t_epoch:
        for data in t_epoch:

            del data.batch_size
            data = Batch.from_data_list([data])  # Stupid, but it adds the batch attribute we need

            x, y_true = _prepare_data_for_model(data, model, device)

            # if model.pool is not None or model.pool_str == "att":
            # We make sure the labels are aggregated per batch
            y_true = global_mean_pool(y_true, data.batch)
            if model.task == "regression":
                y_true = y_true[:, 1]
            y_pred = model(x)

            y_preds.append(y_pred.cpu().detach().numpy())
            y_trues.append(y_true.cpu().detach().numpy())

            loss = criterion(y_pred, y_true)

            # Save loss
            test_loss += loss.item()
            if model.task == "classification":
                test_acc += float(sum(y_pred.argmax(dim=1) == y_true.argmax(dim=1)))
            test_samples += len(y_true)

            yp_cat = np.concatenate(y_preds)
            yt_cat = np.concatenate(y_trues)
            if model.task == "classification":
                yp_cat_argmax = yp_cat.argmax(axis=1)
                yt_cat_argmax = yt_cat.argmax(axis=1)
            elif model.task == "regression":
                yp_cat_argmax = yp_cat
                yt_cat_argmax = yt_cat

            if model.task == "classification":
                if len(y_preds) > 1 and yt_cat_argmax.mean() != 1 and yt_cat_argmax.mean() != 0:

                    # Set preliminary average loss in the progress bar
                    t_epoch.set_postfix(
                        loss=f"{test_loss/test_samples:.3f}",
                        acc=f"{accuracy_score(yt_cat_argmax, yp_cat_argmax):.3f}",
                        f1=f"{f1_score(yt_cat_argmax, yp_cat_argmax):.3f}",
                        prec=f"{precision_score(yt_cat_argmax, yp_cat_argmax, zero_division=0):.3f}",
                        rec=f"{recall_score(yt_cat_argmax, yp_cat_argmax):.3f}",
                        auc=f"{roc_auc_score(yt_cat, yp_cat):.3f}",
                        bias=f"{(yp_cat_argmax == 1).mean():.3f}",  # How many are predicted as AD
                    )

            elif model.task == "regression":
                if len(y_preds) > 1:
                    t_epoch.set_postfix(
                        loss=f"{test_loss/test_samples:.3f}",
                        mse=f"{mean_squared_error(yt_cat, yp_cat):.3f}",
                        r2=f"{r2_score(yt_cat, yp_cat):.3f}",
                        mae=f"{mean_absolute_error(yt_cat, yp_cat):.3f}",
                    )

            gc.collect()
            torch.cuda.empty_cache()

    if model.task == "regression":

        # calculate the correlation between the true and predicted values
        r = np.corrcoef(yt_cat, yp_cat)[0, 1]

        plt.figure()
        plt.scatter(yt_cat, yp_cat)
        plt.title(f"True vs predicted, epoch={epoch} (test set), r={r:.3f}")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.show(block=False)

    elif model.task == "classification":
        # calculate confusion matrix
        cm = confusion_matrix(yt_cat_argmax, yp_cat_argmax)
        cm_df = pd.DataFrame(cm, index=["CT", "AD"], columns=["CT", "AD"])
        print(cm_df)

    return test_loss / test_samples, test_acc / test_samples


def generate_embeddings(
        adata: ad.AnnData, 
        model, 
        device, 
        hp: dict,
        args,
        split_i: int, 
        test_donors, train_donors, 
        save_attention: bool = False) -> ad.AnnData:
    # Generate the embeddings for the entire dataset

    results_cell = {
        "h_1": dict(),
        "h_2": dict(),
        "att_pool": dict(),
    }

    adata.uns[f"h_graph_f{split_i}"] = dict()
    adata.uns[f"y_pred_graph_f{split_i}"] = dict()
    adata.uns["att_idx"] = dict()
    adata.uns[f"att_1_f{split_i}_pos"] = dict()
    adata.uns[f"att_1_f{split_i}_val"] = dict()
    adata.uns[f"att_2_f{split_i}_pos"] = dict()
    adata.uns[f"att_2_f{split_i}_val"] = dict()
    
    # Finally, we store the test/train donors
    adata.uns[f"train_donors_f{split_i}"] = train_donors
    adata.uns[f"test_donors_f{split_i}"] = test_donors
    adata.obs[f"train_donor_f{split_i}"] = adata.obs["Donor ID"].isin(train_donors)
    adata.obs[f"test_donor_f{split_i}"] = adata.obs["Donor ID"].isin(test_donors)

    for donor_id in tqdm(adata.obs["Donor ID"].unique(), desc="Generating embeddings"):
        donor_adata = adata[adata.obs["Donor ID"] == donor_id, :]
        donor_data = next(iter(PygDataLoader(
            dataset=GraphDataset(adata=donor_adata, test=True),
            batch_size=1,
            shuffle=False,
        )))
        donor_data = donor_data.to(device)

        with torch.no_grad():
            y_pred_graph, embeddings, attention_scores = model.forward_with_embeddings(donor_data, ret_att=True)

        h_1, h_2, _, h_graph = embeddings
        att_1, att_2, att_pool = attention_scores

        y_pred_graph = y_pred_graph.cpu().detach().numpy()
        h_1 = h_1.cpu().detach().numpy()
        h_2 = h_2.cpu().detach().numpy()
        h_graph = h_graph.cpu().detach().numpy()
        if att_pool is not None:
            att_pool = att_pool.cpu().detach().numpy()

        for i, key in enumerate(donor_adata.obs.index):
            results_cell["h_1"][key] = h_1[i]
            results_cell["h_2"][key] = h_2[i]
            if att_pool is not None:
                results_cell["att_pool"][key] = att_pool[i]
            else:
                results_cell["att_pool"][key] = None
                
        adata.uns[f"h_graph_f{split_i}"][str(donor_id)] = h_graph
        adata.uns[f"y_pred_graph_f{split_i}"][str(donor_id)] = y_pred_graph

        del h_1, h_2, h_graph, y_pred_graph, embeddings, attention_scores
        gc.collect()
        torch.cuda.empty_cache()

        if save_attention:  # Store them in a dict for now, because combining them is hard
            att1_pos, att1_vals = att_1
            att2_pos, att2_vals = att_2
            att1_pos = att1_pos.cpu().detach().numpy()
            att2_pos = att2_pos.cpu().detach().numpy()
            att1_vals = att1_vals.cpu().detach().numpy()
            att2_vals = att2_vals.cpu().detach().numpy()
            att_1 = (att1_pos, att1_vals)
            att_2 = (att2_pos, att2_vals)
            adata.uns["att_idx"][str(donor_id)] = donor_adata.obs.index.tolist()
            adata.uns[f"att_1_f{split_i}_pos"][str(donor_id)] = att1_pos
            adata.uns[f"att_2_f{split_i}_pos"][str(donor_id)] = att2_pos
            adata.uns[f"att_1_f{split_i}_val"][str(donor_id)] = att1_vals
            adata.uns[f"att_2_f{split_i}_val"][str(donor_id)] = att2_vals

    adata.obsm[f"h_1_f{split_i}"] = np.array([results_cell["h_1"][key] for key in adata.obs.index])
    adata.obsm[f"h_2_f{split_i}"] = np.array([results_cell["h_2"][key] for key in adata.obs.index])
    if att_pool is not None:
        adata.obsm[f"att_pool_f{split_i}"] = np.array([results_cell["att_pool"][key] for key in adata.obs.index])

    del results_cell
    gc.collect()
    torch.cuda.empty_cache()

    return adata


def generate_name(model) -> str:
    return f"{datetime.now():%Y-%m-%d-%H-%M-%S}_{model.name}"


def save_results(model: Union[CellGAT, Linear], adata: ad.AnnData) -> ad.AnnData:

    # For each cell we save whether it was part of the test or train set.
    adata.obs["train_set"] = np.select(
        condlist=[
            adata.obs["Donor ID"].isin(adata.uns["test_donors"]),
            adata.obs["Donor ID"].isin(adata.uns["train_donors"]),
        ],
        choicelist=["Test", "Train"]
    )

    # If we do it on the cpu, we can do it all at once
    model = model.cpu()
    model.eval()
    with torch.no_grad():
        adata = model.calc_embeddings(adata)

    # Calculate the accuracies per cell, we round (argmax) the prediction for hard assignment
    y_pred = adata.obsm["y_pred"]
    y = adata.obs["y"]
    y_pred_soft = y_pred[:, 1]
    y_pred_hard = y_pred.argmax(axis=1)
    adata.obs["predicted_label"] = y_pred_hard
    adata.obs["accuracy"] = 1 - np.abs(y_pred_hard - y)
    adata.obs["accuracy_soft"] = 1 - np.abs(y_pred_soft - y)

    # Calculate number of true positives, false positives, false negatives
    # That are required for the precision, recall and f1 score
    adata_test = adata[adata.obs["train_set"] == "Test"].copy()
    pos = adata_test[adata_test.obs["Label"] == "AD"]
    neg = adata_test[adata_test.obs["Label"] == "CT"]
    n_tp = pos.obs["accuracy"].mean() * pos.shape[0]
    n_fp = (1 - neg.obs["accuracy"].mean()) * neg.shape[0]
    n_fn = (1 - pos.obs["accuracy"].mean()) * pos.shape[0]
    del pos, neg, adata_test

    # Calculate precision, recall, f1
    precision = n_tp / (n_tp + n_fp)
    recall = n_tp / (n_tp + n_fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    adata.uns["performance"]["precision"] = precision
    adata.uns["performance"]["recall"] = recall
    adata.uns["performance"]["f1"] = f1

    # TODO: ROC curve
    # TODO: AUC score

    return adata


def report_results(adata: ad.AnnData) -> None:

    adata_train = adata[adata.obs['train_set'] == 'Train']
    adata_test = adata[adata.obs['train_set'] == 'Test']

    # Calculate balances of the full set, and both train and test set
    def balance(adata: ad.AnnData) -> list[float]:
        return list(adata.obs['Label'].value_counts() / adata.shape[0])

    print("Hyper parameters:")
    for key in adata.uns["hyper_parameters"]:
        print(f" - {key}: {adata.uns['hyper_parameters'][key]}")
    print()
    print("Dataset information:")
    print(f" - no. cells: {adata.shape[0]}")
    print(f" - no. genes: {adata.shape[1]}")
    print(f" - no. donors: {len(adata.obs['Donor ID'].unique())}")
    print(f" - set balance: {balance(adata)}")
    print()
    print(" - Train set:")
    print("    - no. cells:", adata_train.shape[0])
    print("    - no. donors:", len(adata.uns['train_donors']))
    print("    - set balance:", balance(adata_train))
    print()
    print(" - Test set:")
    print("    - no. cells:", adata_test.shape[0])
    print("    - no. donors:", len(adata.uns['test_donors']))
    print("    - set balance:", balance(adata_test))
    print()
    print("------")
    print()
    print("Performance:")
    for key in adata.uns["performance"]:
        print(f" - {key}: {adata.uns['performance'][key]}")
    print()


def sample_weights(adata: ad.AnnData) -> list[float]:
    # We calculate the weight for each sample,
    #   so it corrects for the class imbalance.
    weights = np.zeros(adata.shape[0])

    weights[adata.obs["Label"] == "AD"] = 1 / \
        adata.obs["Label"].value_counts()["AD"]
    weights[adata.obs["Label"] == "CT"] = 1 / \
        adata.obs["Label"].value_counts()["CT"]

    return weights


def _prepare_loader(train_adata, test_adata, hp):

    if hp["model_type"] == "GAT":
        # The train data is loaded in batches from different donors, to provide regularization,
        # and make sure the data fits in GPU memory.
        train_loader = PygNeighborLoader(
            data=GraphDataset(adata=train_adata, test=False)[0],
            num_neighbors=[15, 15],
            batch_size=hp["batch_size"],
            directed=False,
            shuffle=True,
            # sampler=WeightedRandomSampler(
            #     weights=sample_weights(train_adata),
            #     num_samples=len(train_adata),
            # ),
            # TODO: try setting num_workers?
        )
        # In order to test, we cannot use the NeighborLoader, because it does not correctly calculate the accuracy,
        # Instead, we need to do a full forward pass, one per graph. If that doesn't fit in memory, consider moving to cpu for this.
        test_loader = PygDataLoader(
            dataset=GraphDataset(adata=test_adata, test=True),
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )
    elif hp["model_type"] == "Linear":
        # As opposed to the GAT model, the linear model does not require a NeighborLoader
        # And can just use a regular pytorch DataLoader for both train and test sets.
        train_loader = TorchDataLoader(
            dataset=Dataset(adata=train_adata),
            batch_size=hp["batch_size"],
            num_workers=3,
            shuffle=True,
            # sampler=WeightedRandomSampler(
            #     weights=sample_weights(train_adata),
            #     num_samples=len(train_adata),
            # ),
        )
        test_loader = TorchDataLoader(
            dataset=Dataset(adata=test_adata),
            batch_size=hp["batch_size"],
            num_workers=2,
        )

    else:
        raise ValueError(f'Unknown model type: {hp["model_type"]}')

    return train_loader, test_loader


def _check_hps(hp: dict) -> bool:
    if hp["model_type"] == "Linear":
        required_hps = ["lr", "wd", "batch_size", "dim_h"]
    elif hp["model_type"] == "GAT":
        required_hps = ["lr", "wd", "batch_size",
                        "dim_h", "dropout", "heads", "self_loops"]
    else:
        raise ValueError(
            f'Cannot check hyperparameters for unknown model type: {hp["model_type"]}')

    valid = True
    for hp_name in required_hps:
        if hp_name not in hp:
            print(
                f'[ERROR] Model type {hp["model_type"]} requires parameter: {hp_name}')
            valid = False
    return valid


def train_model(
    adata: ad.AnnData,

    # Model and training hyperparameters
    hp: dict,

    # # Training settings
    n_epochs: int,
    test_interval: int,
    split_seed: int,
    max_batches: int = None,

    # Other settings
    verbose: bool = False,
) -> tuple[ad.AnnData, Union[CellGAT, Linear]]:

    exp_start = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if verbose:
        print(f"Experiment started at {exp_start}")

    # First, we check if all required hyper parameters are provided
    if not _check_hps(hp):
        print("Missing hyper parameters, aborting training.")
        exit(-1)
    elif verbose:
        print("Model parameters verified!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For the GAT model, we do the test on cpu,
    # because it doesn't fit in gpu memory
    test_device = "cpu" if hp["model_type"] == "GAT" else device
    test_device = device

    # Load the dataset
    if verbose:
        print("Splitting dataset...")
    hp["dim_in"] = adata.shape[1]  # Derive dim_in from the number of genes
    adata.uns["hyper_parameters"] = hp
    adata.uns["exp_start"] = exp_start

    # Split the dataset
    split_ratio = 0.8
    _tup = adata_train_test_split(adata, split_ratio, split_seed)
    train_adata, test_adata, train_donors, test_donors = _tup
    adata.uns["train_donors"] = train_donors
    adata.uns["test_donors"] = test_donors
    del _tup

    train_loader, test_loader = _prepare_loader(
        train_adata,
        test_adata,
        hp,
    )

    if verbose:
        print("Done splitting dataset")

    # Define the model
    if hp["model_type"] == "Linear":
        model = Linear(
            dim_in=hp["dim_in"],
            dim_h=hp["dim_h"],
            dropout=hp["dropout"] if "dropout" in hp else None,
        )
    elif hp["model_type"] == "GAT":
        model = CellGAT(
            dim_in=hp["dim_in"],
            dim_h=hp["dim_h"],
            dropout=hp["dropout"],
            heads=hp["heads"],
            self_loops=hp["self_loops"],
        )
    else:
        raise ValueError(f'Unknown model type: {hp["model_type"]}')

    model = model.to(device)
    model.train()

    # Initialize optimizer. The choice for Adam is quite arbitrary, try others later?
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp["lr"],
        weight_decay=hp["wd"],
    )

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    n_batches = train_adata.n_obs // hp["batch_size"] + 1
    if verbose:
        print(f"n_batches for training data = {n_batches}")
    if max_batches != -1:
        if verbose:
            print(f"but using max_batches = {max_batches} per epoch")
        n_batches = max_batches

    for epoch in range(n_epochs):

        train_loss, train_acc = _train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, n_batches=n_batches)

        if test_interval != -1 and epoch % test_interval == 0:
            with torch.no_grad():
                model = model.to(test_device)
                test_loss, test_acc = _test_epoch(
                    model, test_loader, criterion, test_device, epoch)
                model = model.to(device)

    # If we don't test every epoch, we do a final test at the end
    if test_interval == -1:
        with torch.no_grad():
            model = model.to(test_device)
            test_loss, test_acc = _test_epoch(
                model, test_loader, criterion, test_device, epoch)
            model = model.to(device)

    del train_adata, train_loader,  # Free up memory

    # Save the performance, TODO: do per epoch instead
    adata.uns['performance'] = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

    adata = save_results(model, adata)

    return adata, model
