"""Evaluate a trained scAGG model on an external validation h5ad.

Loads a model saved by train_full.py (full model object via torch.save),
runs inference per donor, and writes performance metrics to JSON.

The validation h5ad is expected to have:
  - a "Donor ID" column in obs
  - the column named by --label-col with values matching --positive-label
    and --negative-label (defaults: "Wang", "AD", "Healthy")
  - optionally, a boolean --intermediate-col flagging donors to exclude
    from metric computation (default: "Wang_intermediate")
"""
import argparse
import gc
import json

import anndata as ad
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_util import generate_embeddings, mem


parser = argparse.ArgumentParser(description="Evaluate a trained scAGG model on an external validation set.")
parser.add_argument("--dataset", type=str, required=True, help="Path to validation dataset (.h5ad)")
parser.add_argument("--model", type=str, required=True, help="Path to saved full model (.pt)")
parser.add_argument("--label-col", type=str, default="Wang", help="obs column with binary label (default: Wang)")
parser.add_argument("--intermediate-col", type=str, default="Wang_intermediate", help="obs column flagging excluded donors")
parser.add_argument("--positive-label", type=str, default="AD", help="Value of label-col treated as positive class")
parser.add_argument("--negative-label", type=str, default="Healthy", help="Value of label-col treated as negative class")
parser.add_argument("--compute-wang", action="store_true", help="Compute Wang labels from raw SeaAD obs columns ('Cognitive Status', 'Braak', 'CERAD score') and write them to --label-col / --intermediate-col before evaluation")
parser.add_argument("--save-embeddings", action="store_true", help="Save embeddings + h5ad output")
parser.add_argument("--save-attention", action="store_true", help="Save attention scores")
parser.add_argument("--output", type=str, default=None, help="Output h5ad path (only used with --save-embeddings)")
parser.add_argument("--output-metrics", type=str, default=None, help="Output JSON metrics path (default: derived from --output or --model)")
parser.add_argument("--verbose", action="store_true")


SEAAD_BRAAK_MAP = {
    "Braak 0": 0, "Braak I": 1, "Braak II": 2, "Braak III": 3,
    "Braak IV": 4, "Braak V": 5, "Braak VI": 6,
}
SEAAD_CERAD_MAP = {  # ROSMAP convention: 1=worst (Frequent), 4=best (Absent)
    "Frequent": 1, "Moderate": 2, "Sparse": 3, "Absent": 4,
}
SEAAD_COGDX_MAP = {"Dementia": 4, "No dementia": 1}


def compute_wang_labels(adata: ad.AnnData, label_col: str, intermediate_col: str,
                        positive_label: str, negative_label: str) -> None:
    """Derive donor-level Wang labels from raw SeaAD columns and write them
    to adata.obs[label_col] and adata.obs[intermediate_col] in place.

    Mirrors the rule used in train_full.py for ROSMAP, but operates on
    SeaAD's string-valued 'Cognitive Status' / 'Braak' / 'CERAD score'.
    """
    required = ["Cognitive Status", "Braak", "CERAD score", "Donor ID"]
    missing = [c for c in required if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"--compute-wang needs obs columns {required}; missing: {missing}")

    donor_df = adata.obs.groupby("Donor ID", observed=True).first()
    cogdx = donor_df["Cognitive Status"].map(SEAAD_COGDX_MAP)
    braaksc = donor_df["Braak"].map(SEAAD_BRAAK_MAP)
    ceradsc = donor_df["CERAD score"].map(SEAAD_CERAD_MAP)

    is_ad = (cogdx == 4) & (braaksc >= 4) & (ceradsc <= 2)
    is_ct = (cogdx == 1) & (braaksc <= 3) & (ceradsc >= 3)

    donor_label = np.where(is_ad, positive_label,
                  np.where(is_ct, negative_label, None))
    donor_label_map = dict(zip(donor_df.index, donor_label))
    donor_intermediate_map = {d: (donor_label_map[d] is None) for d in donor_df.index}

    adata.obs[label_col] = adata.obs["Donor ID"].map(donor_label_map)
    adata.obs[intermediate_col] = adata.obs["Donor ID"].map(donor_intermediate_map).astype(bool)

    n_ad = int(is_ad.sum())
    n_ct = int(is_ct.sum())
    n_int = int((~is_ad & ~is_ct).sum())
    print(f"  computed Wang labels: {n_ad} {positive_label}, {n_ct} {negative_label}, {n_int} intermediate (out of {len(donor_df)} donors)")


def calc_performance(adata: ad.AnnData, donors: list, donor_y_true: dict) -> dict:
    """Compute classification metrics from generate_embeddings output."""
    y_true = np.array([donor_y_true[d] for d in donors], dtype=int)
    y_pred = np.concatenate([adata.uns["y_pred_graph_f0"][str(d)] for d in donors])
    y_pred_hard = np.argmax(y_pred, axis=1)

    return {
        "accuracy": accuracy_score(y_true, y_pred_hard),
        "precision": precision_score(y_true, y_pred_hard),
        "recall": recall_score(y_true, y_pred_hard),
        "f1": f1_score(y_true, y_pred_hard),
        "roc_auc": roc_auc_score(y_true, y_pred[:, 1]),
        "confusion_matrix": confusion_matrix(y_true, y_pred_hard).tolist(),
        "n_donors": int(len(donors)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
    }


def main():
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset}")
    adata = ad.read_h5ad(filename=args.dataset)
    print(f"  shape={adata.shape}, donors={adata.obs['Donor ID'].nunique()}")

    if args.compute_wang:
        compute_wang_labels(adata, args.label_col, args.intermediate_col,
                            args.positive_label, args.negative_label)

    if args.label_col not in adata.obs.columns:
        raise ValueError(
            f"--label-col '{args.label_col}' not found in obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    # Build the binary y label.
    label = adata.obs[args.label_col]
    y_map = {args.positive_label: 1, args.negative_label: 0}
    adata.obs["y"] = label.map(y_map).astype(float)

    # Donors to score on: have a non-NaN y AND are not flagged as intermediate.
    eval_mask = adata.obs["y"].notna()
    if args.intermediate_col in adata.obs.columns:
        intermediate = adata.obs[args.intermediate_col].astype(bool)
        eval_mask &= ~intermediate
        print(f"  excluding {intermediate.sum()} cells flagged as {args.intermediate_col}")
    else:
        print(f"  warning: '{args.intermediate_col}' not in obs; nothing excluded as intermediate")

    adata_full = adata
    adata_eval = adata[eval_mask].copy()

    eval_donors = adata_eval.obs["Donor ID"].unique().tolist()
    print(f"  scoring on {len(eval_donors)} donors (out of {adata_full.obs['Donor ID'].nunique()} total)")
    print(f"  class balance on eval set: {adata_eval.obs['y'].value_counts().to_dict()}")

    # Donor-level true labels, taken from the (NaN-free) eval slice. Used for
    # metric computation independently of whatever y values end up on the
    # full adata that may also pass through generate_embeddings.
    donor_y_true = adata_eval.obs.groupby("Donor ID", observed=True)["y"].first().astype(int).to_dict()

    device = "cpu"  # eval is small enough; safest default

    print(f"Loading model from {args.model}")
    model = torch.load(args.model, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    # The saved model has its training-time means/stds baked in as Parameters,
    # so we don't recompute them on the validation set.
    if hasattr(model, "means") and model.means is not None:
        if model.means.shape[1] != adata_full.shape[1]:
            raise ValueError(
                f"Gene dim mismatch: model expects {model.means.shape[1]} genes, "
                f"dataset has {adata_full.shape[1]}"
            )

    # Run inference. We pass adata_full so embeddings are also generated for
    # intermediate donors when --save-embeddings is set.
    target = adata_full if args.save_embeddings else adata_eval
    target = generate_embeddings(
        target, model, device, {}, args, 0,
        eval_donors,  # "test_donors"
        [],            # "train_donors"
        save_attention=args.save_attention,
    )

    perf = calc_performance(target, eval_donors, donor_y_true)
    print("\nPerformance on validation set:")
    for k, v in perf.items():
        print(f"  {k}: {v}")

    # Resolve output paths
    metrics_out = args.output_metrics
    if metrics_out is None:
        if args.output is not None:
            metrics_out = args.output.replace(".h5ad", "_metrics.json")
        else:
            metrics_out = args.model.replace(".pt", "_eval_metrics.json")

    metrics = {
        "dataset": args.dataset,
        "model": args.model,
        "args": vars(args),
        "performance": perf,
    }
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nWrote metrics to {metrics_out}")

    if args.save_embeddings or args.save_attention:
        if args.output is None:
            raise ValueError("--save-embeddings requires --output")
        target.uns["perf_eval"] = perf
        print(f"Writing h5ad to {args.output}")
        target.write_h5ad(filename=args.output, compression="gzip")

    del model, target
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mem("after eval")
    print("\nDone.")


if __name__ == "__main__":
    main()
