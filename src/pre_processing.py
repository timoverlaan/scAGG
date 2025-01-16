"""
This file contains the preprocessing pipeline for the data, including the following steps:
- Load and parse the donor metadata
- Load the cell expression data
- Downsample the data (optional)
- Select highly variable genes (seurat v1 or v3)
- Add donor metadata columns to the AnnData
- Calculate PCA
- Build KNN graphs per donor

The output of this script is a set of AnnData files, that can be used for training the GNNs.
"""

import os
import re
from datetime import datetime
from typing import Literal, Tuple
import requests
import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scanpy
from tqdm import tqdm

import env


REFERENCE_DONORS = ["H18.30.001", "H18.30.002", "H19.30.001", "H19.30.002", "H200.1023"]


parser = argparse.ArgumentParser(description='Data pre-processing pipeline')

parser.add_argument('--n_top_genes', type=int, default=1000,
                    help='Number of top genes to select')

parser.add_argument('--downsample', action='store_true', default=False,
                    help='Whether to downsample the data')

parser.add_argument('--inc_ref_donors', action='store_true', default=False,
                    help='Whether to include the (5) reference donors in the data')

parser.add_argument('--gene_selection', type=str, default="seurat_v3",
                    help='Gene selection method to use. Options: [seurat_v1, seurat_v3]')

parser.add_argument('--k_neighbors', type=int, default=30,
                    help='Number of neighbors to use for KNN graph construction')

parser.add_argument('--stringdb', action='store_true', default=False,
                    help='Whether to use only STRINGdb (protein-coding) genes')


def load_donor_data(filter_other: bool = True) -> pd.DataFrame:
    """
    Load and parse the donor data csv file.

    Args:
        path (str): full path to the donor data csv file

    Returns:
        pd.DataFrame: the parsed donor data
    """

    METADATA_URL = "https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/f8/ba/f8ba2fc7-24c3-4409-8915-f78ef65eacc9/sea-ad_cohort_donor_metadata_082222.xlsx"

    path = env.get("METADATA")
    if not os.path.exists(path):

        response = requests.get(METADATA_URL)
        with open(path, "wb") as f:
            f.write(response.content)

    assert os.path.exists(path)
    df = pd.read_excel(path, index_col="Donor ID")

    col_diagnosis = []  # The new column to be constructed
    for donor in df.index:
        diagnosis = [
            re.search(r'choice=(.*?)\)', col)[1]
            for col in filter(
                lambda col_name: "Consensus Clinical Dx" in col_name, df.keys()
            )
            if df.at[donor, col] == "Checked"
        ]

        # If diagnosis is "Other", there is an additional column that stores a note
        if "Other" in diagnosis:
            diagnosis[diagnosis.index(
                "Other")] = f"Other ({df.at[donor, 'If other Consensus dx, describe']})"
        diagnosis = ", ".join(diagnosis)  # Stringify

        col_diagnosis.append(diagnosis)
    df["Diagnosis"] = col_diagnosis

    # Next, we add a new column, that contains the labels (AD/CT/Other)
    df['Label'] = np.select(
        condlist=[
            df['Diagnosis'].str.contains('Alzheimer'),
            df['Diagnosis'].str.contains('Control')
        ],
        choicelist=["AD", "CT"],
        default="Other"
    )

    if filter_other:
        # We remove all the donors with label "Other"
        df = df[df["Label"] != "Other"]

    # Drop the columns we've parsed and don't need anymore
    columns_to_drop = [
        'If other Consensus dx, describe',
        *list(filter(
            lambda col_name: "Consensus Clinical Dx" in col_name, df.keys()
        )),
    ]

    return df.drop(columns=columns_to_drop)


def downsample(adata: ad.AnnData, ddata: pd.DataFrame) -> Tuple[pd.Series, ad.AnnData]:
    """
    Downsample the data, such that:
    - Donors are removed, if their cellcount is less than one standard deviation smaller than the average
    - Donors are downsampled, if their cellcoutn is more than one standard deviation larger than the average

    Args:
        adata (ad.AnnData): The cell expression data, which should include the column "Donor ID"
        ddata (pd.DataFrame): The donor metadata, which should include the column "cellcount"

    Returns:
        pd.Series: a binary mask, that can be used to slice the adata
        pd.DataFrame: the donor metadata, with new column "downsampled", that indicates which donors were downsampled
            it also doesn't contain the rows of the donors that were removed due to low cell counts
    """
    # Add a column with cell counts per donor
    donor_cell_counts = adata.obs["Donor ID"].value_counts()
    ddata["cellcount"] = [donor_cell_counts[donor] for donor in ddata.index]

    # Find donors that have too few / many cells
    avg = ddata["cellcount"].mean()
    std = ddata["cellcount"].std()

    # We remove the donors with too few cells
    ddata = ddata[ddata["cellcount"] > avg - std].copy()
    ddata["downsampled"] = ddata["cellcount"] > avg + std
    mask = adata.obs["Donor ID"].isin(list(ddata.index))

    # For each donor with too many cells, we downsample
    for donor, _ in ddata[ddata["downsampled"]]["cellcount"].items():
        donor_mask = adata.obs["Donor ID"] == donor
        donor_cells_idx = np.arange(adata.shape[0])[donor_mask]
        chosen_cells_idx = np.random.choice(
            donor_cells_idx, size=int(avg), replace=False)
        mask[donor_mask] = False
        mask.iloc[chosen_cells_idx] = True

    return mask, ddata


def select_hvgs(adata: ad.AnnData, method: Literal["seurat", "seurat_v3"], n_top_genes: int) -> None:
    if args.gene_selection == "seurat_v3":
        # This one can also run backed
        scanpy.pp.highly_variable_genes(
            adata,
            layer="UMIs",
            flavor="seurat_v3",
            n_top_genes=args.n_top_genes,
            inplace=True,
            subset=True
        )
    elif args.gene_selection == "seurat":
        scanpy.pp.highly_variable_genes(
            adata,
            flavor="seurat",
            n_top_genes=args.n_top_genes,
            inplace=True,
            subset=True
        )
    else:
        raise NotImplementedError(
            f"ERROR: gene selection '{args.gene_selection}' not implemented.")


def add_donor_metadata(adata: ad.AnnData, ddata: pd.DataFrame) -> None:
    """
    Adds columns of ddata that are missing in adata to adata. (in-place)

    Args:
        adata (ad.AnnData)
        ddata (pd.DataFrame)
    """
    missing_keys = [key for key in ddata.keys() if key not in adata.obs.keys()]

    # First, we pad the donor data with None values for the reference donors
    for donor in REFERENCE_DONORS:
        ddata.loc[donor, :] = None
        ddata.loc[donor, "Label"] = "CT"
        ddata.loc[donor, "Diagnosis"] = "Reference"

    # Expand the donor metadata to match the adata
    df_expanded = pd.DataFrame(
        index=adata.obs_names,
        columns=missing_keys
    )
    for donor in tqdm(ddata.index, desc="Expanding donor metadata"):
        if donor in adata.obs["Donor ID"].unique():
            print("Merging donor: ", donor)
            donor_slice = adata[adata.obs["Donor ID"] == donor]
            donor_metadata = ddata.loc[donor][missing_keys]
            print(f"Donor missing metadata: {donor_metadata.shape}")
            for obs_name in donor_slice.obs_names:
                df_expanded.loc[obs_name] = donor_metadata
            # df_expanded.loc[donor_slice.obs_names] = donor_metadata

    # Add the expanded donor metadata to the adata
    adata.obs = pd.concat([adata.obs, df_expanded], axis=1)

    # ddata.loc[adata.obs["Donor ID"]]
    # for key in missing_keys:
    #     adata.obs[key] = df_expanded[key].values


def clean_columns(adata: ad.AnnData, _ddata: pd.DataFrame) -> None:
    """
    TODO: still have to implement this sometime (but it's not urgent)

    Removes any columns that are not useful for training the GNNs or analysing the results.
    Should also combine columns that are redundant (like race, which should be a categorical column)

    Args:
        adata (ad.AnnData)
        ddata (pd.DataFrame)

    Returns:
        None: because the operations are in-place

    """
    del adata.layers["UMIs"]  # Not needed anymore, and doubles the file size


def build_knn_graphs(adata: ad.AnnData, k_neighbors: int) -> ad.AnnData:
    """
    Use scanpy to make a KNN graph per donor.
    The AnnData object is first sliced per donor, then the KNN graph is calculated for each slice.
    Finally, the slices are concatenated into a single AnnData object.

    Args:
        adata (ad.AnnData)
        k_neighbors (int)

    Returns:
        ad.AnnData
    """

    slices = []
    for donor in tqdm(adata.obs["Donor ID"].unique(), desc="Building donor KNN graphs"):
        donor_slice = adata[adata.obs["Donor ID"] == donor]
        scanpy.pp.neighbors(
            donor_slice, n_neighbors=k_neighbors, use_rep="X_pca")
        slices.append(donor_slice)

    return ad.concat(
        slices,
        merge="same",
        uns_merge="same",
        label="all",
        index_unique="-",
        pairwise=True,
    )


if __name__ == "__main__":

    start = datetime.now()
    print("Starting pre-processing pipeline...")

    args = parser.parse_args()
    output_path = env.get("PROCESSED_DATA")

    gene_suffix = {"seurat": "_s1", "seurat_v3": "_s3"}[args.gene_selection]
    downsample_suffix = "_ds" if args.downsample else ""
    stringdb_suffix = "_sdb" if args.stringdb else ""

    top_genes_file = f"{output_path}/adata_top{args.n_top_genes}{stringdb_suffix}{gene_suffix}{downsample_suffix}.h5ad"

    if not os.path.exists(top_genes_file):

        assert args.gene_selection in {"seurat", "seurat_v3"}
        assert os.path.exists(env.get("RAW_DATA"))

        # Remove trailing slash if present
        if output_path[-1] == "/":
            output_path = output_path[:-1]

        # First load and parse donor metadata
        print("Loading donor metadata...")
        ddata = load_donor_data()
        print(f"Donor metadata loaded, shape={ddata.shape}")

        included_donors = list(ddata.index)
        if args.inc_ref_donors:
            included_donors += REFERENCE_DONORS
        print(f"Included {len(included_donors)} donors of the data")
        
        
        # Then cell expression data (requires ~200GB memory)
        print("Loading AnnData to memory...")
        adata = ad.read_h5ad(filename=env.get("RAW_DATA"))
        print(f"AnnData loaded, shape={adata.shape}")

        print("Adding donor metadata columns to AnnData...")
        add_donor_metadata(adata, ddata)

        if args.downsample:
            # TODO: this is old code, and it doesn't include the reference donors
            downsample_mask, ddata = downsample(adata, ddata)
            adata_ds = adata[downsample_mask].copy()
        else:
            adata_ds = adata[adata.obs["Donor ID"].isin(included_donors)].copy()

        del adata  # Free up memory
        adata = adata_ds

        if args.stringdb:
            print("Filtering out non-STRINGdb genes...")
            from dataset.stringdb import import_string_db
            genes, _edges = import_string_db(adata.var_names)
            adata = adata[:, genes].copy()
            print(f"Filtered out non-STRINGdb genes, shape={adata.shape}")

        # Next, we run highly variable gene selection, this also takes quite some memory
        print(f"Selecting highly variable genes with method={args.gene_selection}...")
        select_hvgs(adata, args.gene_selection, args.n_top_genes)

        # We add a column 'y', that is 1.0 for AD, and 0.0 for CT
        adata.obs["y"] = np.where(adata.obs["Label"] == "AD", 1.0, 0.0)

        clean_columns(adata, ddata)
        ddata.to_csv("donor_data.csv")

        # First we calculate PCA on the full dataset
        print("Caclulating PCA...")
        scanpy.pp.pca(adata, n_comps=50)
        assert "X_pca" in adata.obsm  # scanpy's PCA should have added this
        print("Finished PCA")

        # Now we save, because from here on we might want to use different k's
        adata.write_h5ad(filename=top_genes_file, compression="gzip")
        print(f"Result was written to {top_genes_file}")

    else:
        # Only load the data that was already processed.
        print(f"Loading processed data to memory from file: {top_genes_file}...")
        adata = ad.read_h5ad(filename=top_genes_file)
        print(f"Loaded processed data, shape={adata.shape}")

    # Now we can build the KNN graphs
    adata_knn = build_knn_graphs(adata, k_neighbors=args.k_neighbors)

    # And save the resulting file
    adata_knn.write_h5ad(filename=top_genes_file.replace(".h5ad", f"_k{args.k_neighbors}.h5ad"), compression="gzip")
    
    end = datetime.now()
    print(f"Took {end-start} to complete.")
