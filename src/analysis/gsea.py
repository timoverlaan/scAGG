import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gseapy as gp

from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.stats import spearmanr


FOLD = 4  # (just pick one, I'll check if it matters, but I don't expect so)


def rank_genes(
        adata: ad.AnnData, 
        celltype: str,
        drop_int: bool = True, 
        correct_only: bool = False,
    ) -> pd.Series:

    # Drop intermediate donors for this part
    if drop_int:
        adata = adata[adata.obs["Label"] != "Other"]  

    # Find which donors were correctly classified in fold
    correct_donors = get_correct_donors(adata)

    if correct_only:
        adata = adata[adata.obs["Donor ID"].isin(correct_donors)]

    # PCA, and plot colored by class
    pca = PCA(n_components=5)
    x_pca = pca.fit_transform(adata.obsm[f"h_2_f{FOLD}"].toarray())
    
    plt.figure()
    df = pd.DataFrame({
        "PCA1": x_pca[:, 0],
        "Color": adata.obs["Label"].values.map({"CT": "Healthy", "AD": "AD"})
    })
    sns.histplot(df, x="PCA1", hue="Color", kde=True, hue_order=["Healthy", "AD"])
    plt.savefig(f"out/results/gsea/pca_colored_by_class_{celltype}.png")

    # ---------

    # We use spearman correlation to find genes that are correlated with the first PCA component.
    # This is better than using Pearson correlation since the gene expression data is not normally distributed, and highly sparse.

    spearman_corrs = []
    skipped_genes = []
    gene_cells = []

    for i, gene in enumerate(tqdm(adata.var_names)):
        gene_expr = adata.X[:, i].toarray().flatten()
        gene_cells.append(gene_expr.nonzero()[0].shape[0])

        # Skip genes with zero variance (actually, they just have all 0's)
        if np.var(gene_expr) == 0:
            skipped_genes.append(gene)
            spearman_corrs.append(0)
            continue

        corr, _ = spearmanr(gene_expr, x_pca[:, 0])
        spearman_corrs.append(corr)

    print(f"Skipped {len(skipped_genes)} genes due to zero variance:")
    print(skipped_genes)

    spearman_corrs = pd.Series(spearman_corrs, index=adata.var_names)

    ax, fig = plt.subplots()
    sns.histplot(spearman_corrs)
    plt.title("Spearman correlation between gene expression and 1st PC")
    plt.savefig(f"out/results/gsea/spearman_correlation_{celltype}.png")

    ax, fig = plt.subplots()
    sns.histplot(gene_cells)
    plt.title("Number of cells with non-zero expression for each gene")
    plt.savefig(f"out/results/gsea/gene_cells_{celltype}.png")

    # Rank the genes by their spearman correlation (absolute value)
    ranking = pd.Series(spearman_corrs, index=adata.var_names)
    ranking = ranking[ranking.abs().sort_values(ascending=False).index]
    return ranking


def run_gsea(ranking: pd.Series, celltype: str, gene_sets: list[str]) -> pd.DataFrame:

    res = gp.prerank(
        rnk=ranking, 
        gene_sets=gene_sets,
        min_size=1,  # (default = 15)
        max_size=10000,  # (default = 500)
        verbose=True,
    ).res2d.copy()

    # Show only significant results, sorted by absolute NES
    res_sig = res[res["FDR q-val"] < 0.05].copy()
    res_sig.insert(0, "abs_NES", res_sig["NES"].abs())
    res_sig.sort_values("NES", ascending=False, inplace=True)
    n_sig = res_sig.shape[0]
    print(f"Found {n_sig} significant terms")

    # Output
    res_sig["Celltype"] = celltype
    res_sig.drop(columns=["Name", "abs_NES"])

    return res_sig



def get_correct_donors(adata):
    predictions = adata.uns[f"y_pred_graph_f{FOLD}"]  # Dictionary with all donors, values are np arrays of length 2 [p(CT), p(AD)]

    correct_donors = []
    for donor in adata.obs["Donor ID"].unique():
        if predictions[f"{donor}"][0, 1] < 0.5:  # p(AD)
            pred_label = "CT"
        else:
            pred_label = "AD"
            
        true_label = adata.obs["Label"][adata.obs["Donor ID"] == donor].values[0]
        if pred_label == true_label:
            correct_donors.append(donor)

    return correct_donors


if __name__ == "__main__":

    print("Reading data...")
    adata = ad.read_h5ad(FILE := "out/results/GAT_2024-05-23-16-24-07_results.h5ad")
    print(f"Loaded {FILE}")
    print(adata)

    CELLTYPES = [
        "Astrocytes",
        "Oligodendrocytes",
        "Excitatory_neurons",
        "Inhibitory_neurons",
        "OPCs",
        "Immune_cells",
        "Vasculature_cells",
    ]

    dfs = {}

    for celltype in CELLTYPES:
        print(f"Running GSEA for {celltype}...")

        if celltype == "Excitatory_neurons":
            adata_ct = adata[adata.obs["Supertype"].isin(
                ["Excitatory_neurons_set1, Excitatory_neurons_set2", "Excitatory_neurons_set3"])]
        else:
            adata_ct = adata[adata.obs["Supertype"] == celltype]

        ranking = rank_genes(adata_ct, celltype)

        df = run_gsea(ranking, celltype, gene_sets=[
            # Disease phenotype related:
            # "DisGeNET",
            # "OMIM_Disease",
            # "OMIM_Expanded",
            # "UK_Biobank_GWAS_v1",
            # "Jensen_DISEASES"
            # "GWAS_Catalog_2023",

            # Brain-specific:
            # "PanglaoDB_Augmented_2021",
            # "Aging_Perturbations_from_GEO_down",
            # "Aging_Perturbations_from_GEO_up",
            # "Allen_Brain_Atlas_10x_scRNA_2021",
            # "Allen_Brain_Atlas_down",
            # "Allen_Brain_Atlas_up",

            # Pathway databases
            "GO_Biological_Process_2023",
            # "GO_Cellular_Component_2023",
            # "GO_Molecular_Function_2023"
            # "GTEx_Aging_Signatures_2021",
            # "HDSigDB_Human_2021",
            "KEGG_2021_Human",
            # "WikiPathways_2024_Human",

            ## ??
            # "Azimuth_Cell_Types_2021",
        ])
        dfs[celltype] = df

    print("Combining all results, and writing to file...")
    df_all = pd.concat(dfs.values())
    df_all.to_csv("out/results/gsea/gsea_all_cts.csv")
    print("Done!")
