import numpy as np
import anndata as ad
import pandas as pd
import torch
import argparse
import gc

from datetime import datetime
from tqdm import tqdm, trange
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.loader import NeighborLoader as PygNeighborLoader
from torch_geometric.loader import NodeLoader, RandomNodeLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import WeightedRandomSampler, DataLoader
# from torchinfo import summary

from train_util import _test_epoch, _train_epoch, generate_embeddings, mem
from dataset.GraphDataset import GraphDataset
from dataset.Dataset import Dataset
from models.CellGAT import CellGAT
from models.NoGraph import NoGraph
from dataset.split import adata_kfold_split


parser = argparse.ArgumentParser(description='Train a model on a dataset.')
parser.add_argument('--dataset', type=str, help='Path to the dataset (.h5ad)')
parser.add_argument('--model', type=str, required=True, help="Path to saved model (.pt) to load for evaluation")
parser.add_argument('--save-attention', action="store_true", help='Save the attention scores?')
parser.add_argument('--save-embeddings', action="store_true", help='Save the embeddings?')
parser.add_argument('--verbose', action="store_true", help='Verbose')
parser.add_argument('--label', type=str, default="cogdx", help='Label type [cogdx, raegan, raegan-no-intermediate, wang]')
parser.add_argument('--batch-stratify-sex', action="store_true", help='Stratify the batches also based on sex, to regress this out')
parser.add_argument('--output', type=str, default=None, help='Output file name for the results')

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


if __name__ == "__main__":

    args = parser.parse_args()

    adata = ad.read_h5ad(filename=args.dataset)
    if args.verbose:
        print(f"Loaded dataset {args.dataset} with {adata.n_obs} cells and {adata.n_vars} genes.")

    # NOTE: below are a bunch of steps that should ideally be moved to a processing script instead.

    # print("All celltypes", adata.obs["cell_type_high_resolution"].unique().tolist())
    # print(adata.obs.keys().tolist())
    # print("Major celltypes", adata.obs["Supertype"].unique().tolist())
    # mic_celltypes = [ct for ct in adata.obs["cell_type_high_resolution"].unique() if "Mic" in ct or "Ast" in ct]
    # print(f"Microglia cell types: {mic_celltypes}")

    # adata.obs["Celltype"] = adata.obs["Supertype"].map({
    #     "Astrocytes": "Astrocytes",
    #     "Excitatory_neurons_set1": "Excitatory_neurons",
    #     "Excitatory_neurons_set2": "Excitatory_neurons",
    #     "Excitatory_neurons_set3": "Excitatory_neurons",
    #     "Immune_cells": "Immune_cells",
    #     "Inhibitory_neurons": "Inhibitory_neurons",
    #     "Oligodendrocytes": "Oligodendrocytes",
    #     "OPCs": "OPCs",
    #     "Vasculature_cells": "Vasculature_cells",
    # })
    
    # adata_tmp = adata[adata.obs["Celltype"] == "OPCs"].copy()
    # del adata
    # adata = adata_tmp

    # For SeaAD, we have to make the metadata match first
    # if "msex" not in adata.obs.columns:  
    #     adata.obs["msex"] = adata.obs["Sex"].map({"Male": 1, "Female": 0})
    # if "cogdx" not in adata.obs.columns:
    #     adata.obs["cogdx"] = adata.obs["Label"].map({"AD": 4, "CT": 1})
    # if "braaksc" not in adata.obs.columns:
    #     adata.obs["braaksc"] = adata.obs["Braak"].map({"Braak 0": 0, "Braak I": 1, "Braak II": 2, "Braak III": 3, "Braak IV": 4, "Braak V": 5, "Braak VI": 6})
    # if "ceradsc" not in adata.obs.columns:
    #     adata.obs["ceradsc"] = adata.obs["CERAD score"].map({"Absent": 4, "Sparse": 3, "Moderate": 2, "Frequent": 1})

    donor_counts = adata.obs["Donor ID"].value_counts()
    # donor_sex = adata.obs.groupby("Donor ID").first()["msex"]

    print("\nArguments:")
    print(args)
    
    exp_start = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_file_base_name = f"{exp_start}_EVAL_{args.label}"
    if args.verbose:
        print(f"Experiment started at {exp_start}")

    test_device = "cpu"

    if args.label in ["braak", "amyloid", "ceradsc", "nft", "tangles", "plaq_n_mf"]:
        task = "regression"
    else:
        task = "classification"

    # Remap the labels, based on provided argument
    if args.label == "reagan":
        # Remap labels based on the NIA-Reagan score
        adata.obs["Label"] = adata.obs["niareagansc"].map({1: "AD", 2: "AD", 3: "CT", 4: "CT"})
    elif args.label == "reagan-no-intermediate":
        # Remap labels based on the NIA-Reagan score
        adata.obs["Label"] = adata.obs["niareagansc"].map({1: "AD", 2: "Other", 3: "CT", 4: "CT"})
    elif args.label == "wang":
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

    elif args.label == "braak":
        adata.obs["Label"] = adata.obs["braaksc"]  # (0-6)


    elif args.label in ["amyloid", "ceradsc", "amyloid", "nft", "tangles", "plaq_n_mf"]:
        # We do regression, because this is a continuous variable
        adata.obs["Label"] = adata.obs[args.label]  # (0-6)

    elif args.label == "cogdx":
        # Remap labels based on the cognitive diagnosis
        adata.obs["Label"] = adata.obs["cogdx"].map({1: "CT", 2: "CT", 3: "CT", 4: "AD", 5: "AD"})

    else:
        adata.obs["Label"] = adata.obs[args.label]  # Use the label as is
    
    donor_labels = adata.obs.groupby("Donor ID").first()["Label"]

    print(f"Using labels: {args.label}")
    print(adata.obs["Label"].value_counts())

    adata.obs["y"] = adata.obs["Wang"].map({"AD": 1, "Healthy": 0, "Intermediate": 1}).astype(int)
    # TODO: not very happy with setting them to 1, but for the performanc this shouldn't matter,
    # because they will be sliced out.

    # If specified, we drop "Other" donors here to make training easier
    adata_full = adata  # Keep this, because we want embeddings for "intermediate" donors as well

    adata = adata[~adata.obs["Wang_intermediate"]].copy()  # Drop the intermediate donors, if specified

    ALL_DONORS = adata.obs["Donor ID"].unique().tolist()

    # Below we incrementally calculate the means and stds for normalization for the training data.
    # Why don't we directly apply it? Because we want to keep the data sparse as much as possible, 
    #   and when standardizing, we lose the zeros.
    # So instead, we calculate the mean and std here, and then apply it in the forward pass of the model.
    print("Calculating means and stds for normalization...")
    sums = np.zeros((1, adata.shape[1]))
    sum_sqs = np.zeros((1, adata.shape[1]))
    chunk_size = 10000
    for i in trange(0, adata.shape[0], chunk_size):

        chunk = adata.X[i:i+chunk_size].todense()
        sums += chunk.sum(axis=0)
        sum_sqs += (np.power(chunk, 2)).sum(axis=0)

    means = sums / adata.shape[0]
    stds = np.sqrt(sum_sqs / adata.shape[0] - np.power(means, 2))
    stds[stds == 0] = 1
    means = torch.tensor(means, dtype=torch.float32)
    stds = torch.tensor(stds, dtype=torch.float32)
    print("Done calculating means and stds for normalization.")

    # The train data is loaded in batches from different donors, to provide regularization,
    # and make sure the data fits in GPU memory.
    # if not args.no_graph:
    test_loader = PygDataLoader(
        dataset=GraphDataset(adata=adata, test=True),
        batch_size=1,
        shuffle=False,
        # num_workers=2,
    )

    if args.verbose:
        print("Train loader ready.")

    # Load the model
    model = torch.load(args.model).to(test_device)
    model.eval()

    # summary(model)

    # Define the loss function
    if task == "regression":
        criterion = torch.nn.MSELoss(reduction='sum')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    gc.collect(), torch.cuda.empty_cache(), mem("after train")
    
    model = model.to(test_device)
    model.eval()

    # Generate the embeddings
    if args.save_embeddings or args.save_attention:
        adata_full = generate_embeddings(adata_full, model, test_device, {}, args, 0, ALL_DONORS, [], save_attention=args.save_attention)
        adata_full.uns[f"perf_test"] = calc_fold_performance(adata_full, 0, ALL_DONORS)

    else:
        adata = generate_embeddings(adata, model, test_device, {}, args, 0, ALL_DONORS, [], save_attention=args.save_attention)
        adata_full.uns[f"perf_test"] = calc_fold_performance(adata, 0, ALL_DONORS)

    del model

    gc.collect(), torch.cuda.empty_cache(), mem("after test/embeddings")
    print(f"\nPerformance:")
    for k, v in adata_full.uns[f"perf_test"].items():
        print(f"    {k}: {v}")
    print()

    # confusion matrix
    print("Confusion matrix:")
    from sklearn.metrics import confusion_matrix
    y_pred = np.array([np.array(v).flatten()[1] for v in adata.uns[f"y_pred_graph_f0"].values()])
    print(y_pred.shape, y_pred)
    y_true = np.array([
        adata_full[adata_full.obs["Donor ID"] == donor_id].obs["y"].mean() for donor_id in adata.uns[f"y_pred_graph_f0"].keys()
    ])

    print(f"y_true: {y_true.shape} --> {y_true[:10]}")
    print(f"y_pred: {len(y_pred)} --> {y_pred[:10]}")
    
    cm = confusion_matrix(y_true, y_pred.round())
    print(cm)
    print()

    # and plot the ROC
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc}")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    if args.save_embeddings or args.save_attention or args.output is not None:
        if args.output is None:
            out_file = f"out/results/{out_file_base_name}_results.h5ad"
        else:
            out_file = args.output
        print("Writing results to disk...")
        adata_full.write_h5ad(filename=out_file, compression="gzip")
        print(f"Done writing results to disk. Filename: {out_file}")

    mem("after EVERYTHING")
    print("\nEverything done!\n")
