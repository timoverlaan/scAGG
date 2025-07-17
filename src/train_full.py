import numpy as np
import anndata as ad
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
parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=0.00005, help='Weight decay')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
parser.add_argument('--dropout', type=float, default=0.75, help='Dropout')
parser.add_argument('--dim', type=int, default=128, help='Hidden dimension')
parser.add_argument('--heads', type=int, default=8, help='Number of attention heads in the GAT moduels')
parser.add_argument('--self-loops', action="store_true", help='Self-loops in the GAT modules')
parser.add_argument('--n-epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--save', action="store_true", help='Save the results? (for large experiments)')
parser.add_argument('--save-attention', action="store_true", help='Save the attention scores?')
parser.add_argument('--save-embeddings', action="store_true", help='Save the embeddings?')
parser.add_argument('--pooling', type=str, default="mean", help='Pooling method (mean, max, median)')
parser.add_argument('--sag', action="store_true", help='Use SAG pooling')
parser.add_argument('--verbose', action="store_true", help='Verbose')
parser.add_argument('--label', type=str, default="cogdx", help='Label type [cogdx, raegan, raegan-no-intermediate, wang]')
parser.add_argument('--no-graph', action="store_true", help='Use the NoGraph baseline model')
parser.add_argument('--batch-stratify-sex', action="store_true", help='Stratify the batches also based on sex, to regress this out')
parser.add_argument('--output', type=str, default=None, help='Output file name for the results')
parser.add_argument('--output-model', type=str, default=None, help='Output file name for the model')


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

    # Model hyper parameters
    hp={
        "model_type": "GAT",
        "lr": args.learning_rate,
        "wd": args.weight_decay,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "dim_in": adata.shape[1],
        "dim_h": args.dim,
        "heads": args.heads,
        "heads2": 4,
        "self_loops": args.self_loops,
        "n_epochs": args.n_epochs,
        "pooling": args.pooling,
    }
    adata.uns["hp"] = hp

    print("Hyperparameters:")
    print(hp)

    print("\nArguments:")
    print(args)
    
    exp_start = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_file_base_name = f"{hp['model_type']}_{exp_start}"
    if args.verbose:
        print(f"Experiment started at {exp_start}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # For the GAT model, we do the test on cpu, because it doesn't fit in gpu memory
    # test_device = "cpu" if hp["model_type"] == "GAT" else device
    # test_device = device
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
    
    if task == "classification":
        # If specified, we drop "Other" donors here to make training easier
        adata_full = adata  # Keep this, because we want embeddings for "intermediate" donors as well
        if "AD" in adata.obs["Label"].unique() and "CT" in adata.obs["Label"].unique():
            adata = adata[adata.obs["Label"].isin(["AD", "CT"])].copy()
            adata.obs["y"] = adata.obs["Label"].map({"AD": 1, "CT": 0})
        if 0 in adata.obs["Label"].unique() and 1 in adata.obs["Label"].unique():
            adata = adata[adata.obs["Label"].isin([0, 1])].copy()
            adata.obs["y"] = adata.obs["Label"]
    else:
        adata_full = adata
        if args.label in ["amyloid", "plaq_n_mf"]:
            adata.obs["y"] = np.log1p(adata.obs["Label"])
        else:
            adata.obs["y"] = adata.obs["Label"]
        adata_slice = adata[adata.obs["y"].isna() == False].copy()
        del adata
        adata = adata_slice

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

    if task == "classification":
        train_bal = adata.obs["y"].value_counts()[1] / adata.n_obs
        print(f"Train balance p(AD)={train_bal:.3f}")

        if args.batch_stratify_sex:
            class_weight = {
                adata.obs["Label"].unique()[0]: { 0: 0, 1: 0, },
                adata.obs["Label"].unique()[1]: { 0: 0, 1: 0, }
            }
            for donor_id in ALL_DONORS:
                class_weight[donor_labels[donor_id]][donor_sex[donor_id]] += 1

        else:  # Only stratify on class label
            class_weight = {
                adata.obs["Label"].unique()[0]: 0,
                adata.obs["Label"].unique()[1]: 0,
            }
            for donor_id in ALL_DONORS:
                class_weight[donor_labels[donor_id]] += 1

        print("Class weights:")
        print(class_weight)

        # We sample according to class prior and correct for graph size,
        #   so large graphs are not sampled from more frequently.
        donor_weights = dict()
        for donor_id in ALL_DONORS:
            if args.batch_stratify_sex:
                # label_weight = 1 / class_weight[donor_labels[donor_id]][donor_sex[donor_id]]
                pass
            else:
                # donor_weights[donor_id] = (1 / donor_counts[donor_id]) * (1 / class_weight[donor_labels[donor_id]])
                donor_weights[donor_id] = (1 / class_weight[donor_labels[donor_id]])
        sample_weights = torch.tensor([donor_weights[donor_id] for donor_id in adata.obs["Donor ID"]], dtype=torch.float32)




    # The train data is loaded in batches from different donors, to provide regularization,
    # and make sure the data fits in GPU memory.
    # if not args.no_graph:
    train_loader = PygNeighborLoader(
        data=GraphDataset(adata=adata, test=False)[0],
        num_neighbors=[0, 0] if args.no_graph and False else [15, 15],
        batch_size=1,  # We manually build the batches in the training loop
        shuffle=False,
        # disjoint=True,  # I want to use this, but it requires pyg-lib, which is not available for windows
        directed=False,
        # subgraph_type="induced",  # or: "directional"  -> required for later versions of PyG
        # sampler=ImbalancedSampler(dataset=torch.tensor(train_adata.obs["y"].values, dtype=torch.long)),
        sampler=WeightedRandomSampler(weights=sample_weights, num_samples=adata.n_obs, replacement=True),
    )

    if args.verbose:
        print("Train loader ready.")

    # Define the model
    if args.no_graph:
        ModelClass = NoGraph
    else:
        ModelClass = CellGAT

    model = ModelClass(
        dim_in=hp["dim_in"],
        dim_h=hp["dim_h"],
        dropout=hp["dropout"],
        heads=hp["heads"],
        heads2=hp["heads2"],
        self_loops=hp["self_loops"],
        SAGrate=0.5,
        pooling=args.pooling,
        sag=args.sag,
        task=task,
        sex_covariate=True,
        means = means,
        stds = stds,
    )

    model = model.to(device)
    model.train()

    # summary(model)

    # Initialize optimizer. The choice for Adam is quite arbitrary, try others later?
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=hp["lr"],
    #     weight_decay=hp["wd"],
    # )

    optimizer = torch.optim.Adagrad(
        model.parameters(),
        lr=hp["lr"],
        weight_decay=hp["wd"],
    )

    # Define the loss function
    if task == "regression":
        criterion = torch.nn.MSELoss(reduction='sum')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    max_batches = 1000
    n_batches = adata.n_obs // hp["batch_size"] + 1
    if args.verbose:
        print(f"n_batches for training data = {n_batches}")
    
    if max_batches != -1:
        if args.verbose:
            print(f"but using max_batches = {max_batches} per epoch")
        n_batches = max_batches
    # Train the model
    for epoch in range(args.n_epochs):
        train_loss, train_acc = _train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, n_batches, hp)

    # Free-up memory
    del train_loader
    gc.collect(), torch.cuda.empty_cache(), mem("after train")

    if args.save:
        if args.output is None:
            model_out_file = f"out/results/{out_file_base_name}_model.pt"
        else:
            model_out_file = args.output_model
            if not model_out_file.endswith(".pt"):
                model_out_file += ".pt"
        print(f"Saving model to {model_out_file} and {model_out_file.replace('.pt', '_statedict.pt')}")
        
        torch.save(model, model_out_file)
        torch.save(model.state_dict(), model_out_file.replace(".pt", "_statedict.pt"))
    
    model = model.to(test_device)
    model.eval()

    # Generate the embeddings
    if args.save_embeddings or args.save_attention:
        adata_full = generate_embeddings(adata_full, model, test_device, hp, args, 0, [], ALL_DONORS, save_attention=args.save_attention)
        adata_full.uns[f"perf_train_f0"] = calc_fold_performance(adata_full, 0, ALL_DONORS)

    del model
    gc.collect(), torch.cuda.empty_cache(), mem("after test/embeddings")

    print("\n=======================")
    print("| Finished all folds! |")
    print("=======================\n")

    # if args.save:
    #     if args.output is None:
    #         out_file = f"out/results/{out_file_base_name}_results.h5ad"
    #     else:
    #         out_file = args.output

    #     print("Writing results to disk...")
    #     adata_full.write_h5ad(filename=out_file, compression="gzip")
    #     print(f"Done writing results to disk. Filename: {out_file}")

    mem("after EVERYTHING")
    print("\nEverything done!\n")
