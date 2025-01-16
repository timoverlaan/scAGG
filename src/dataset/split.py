import anndata as ad
from typing import Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# Constants that are used to access the columns of the AnnData object
DONOR_COL = "Donor ID"
LABEL_COL = "Label"


def adata_kfold_split(adata: ad.AnnData, n_splits: int, seed: None, task: str = "classification") -> Tuple[ad.AnnData, ad.AnnData]:

    if seed is not None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        skf = StratifiedKFold(n_splits=n_splits)

    if task == "regression":
        # cannot stratify
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Get the unique donors and their corresponding class labels
    donors = adata.obs[DONOR_COL].unique()
    donor_to_class = adata.obs.drop_duplicates(
        DONOR_COL).set_index(DONOR_COL)[LABEL_COL]
    
    # Split the donors into train and test sets, stratified by class
    return [
        (
            donors[train_donors], 
            donors[test_donors], 
            # adata[adata.obs[DONOR_COL].isin(donors[train_donors]), :].copy(), 
            # adata[adata.obs[DONOR_COL].isin(donors[test_donors]), :].copy(),
        )
        for train_donors, test_donors in skf.split(donors, donor_to_class.loc[donors])
    ]


def adata_train_test_split(adata: ad.AnnData, ratio: float, seed: int) -> Tuple[ad.AnnData, ad.AnnData]:
    """Split an AnnData object into train and test sets, in a class-stratified manner.

    Args:
        adata (ad.AnnData): The AnnData object to split.
        ratio (float): The ratio of the train set.
        seed (int): Random generator seed.

    Returns:
        Tuple[ad.AnnData, ad.AnnData]: The train and test sets.
    """

    # Ensure that the ratio is a valid proportion
    assert 0.0 < ratio < 1.0, 'Ratio must be between 0 and 1'

    # Get the unique donors and their corresponding class labels
    donors = adata.obs[DONOR_COL].unique()
    donor_to_class = adata.obs.drop_duplicates(
        DONOR_COL).set_index(DONOR_COL)[LABEL_COL]

    # Split the donors into train and test sets, stratified by class
    train_donors, test_donors = train_test_split(
        donors,
        train_size=ratio,
        random_state=seed,
        stratify=donor_to_class.loc[donors],
    )

    # Split the AnnData object based on the donors
    adata_train = adata[adata.obs[DONOR_COL].isin(train_donors), :].copy()
    adata_test = adata[adata.obs[DONOR_COL].isin(test_donors), :].copy()

    return adata_train, adata_test, train_donors, test_donors


if __name__ == "__main__":
    # A simple testing setup

    adata = ad.read_h5ad(filename="data/adata_top1000_s3_ds.h5ad")
    train, test = adata_train_test_split(adata, ratio=0.8, seed=42)

    all_donors = adata.obs[DONOR_COL].unique()
    train_donors = train.obs[DONOR_COL].unique()
    test_donors = test.obs[DONOR_COL].unique()

    # Assert that the train and test sets are disjoint
    for d in train_donors:
        assert d not in test_donors
    for d in test_donors:
        assert d not in train_donors

    # Assert that all donors are in either the train or test set
    for d in all_donors:
        assert d in train_donors or d in test_donors

    # Assert that all cells are used
    assert adata.shape[0] == (train.shape[0] + test.shape[0])

    # Check if the classes are more or less stratified
    print("Original balance:")
    print(adata.obs["Label"].value_counts() / adata.shape[0])
    print()
    print("Train balance:")
    print(train.obs["Label"].value_counts() / train.shape[0])
    print()
    print("Test balance:")
    print(test.obs["Label"].value_counts() / test.shape[0])


    # Test the k-fold setup
    for i, (train_donors, test_donors, train, test) in enumerate(adata_kfold_split(adata, n_splits=5, seed=42)):
        print(f"Split {i}")
        print(f"Train donors: {train_donors}")
        print(f"Train shape: {train.shape}")
        print(f"Train balance: {train.obs[LABEL_COL].value_counts() / train.shape[0]}")
        print(f"Test donors: {test_donors}")
        print(f"Test shape: {test.shape}")
        print(f"Test balance: {test.obs[LABEL_COL].value_counts() / test.shape[0]}")
        print()