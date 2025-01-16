import anndata as ad
import pandas as pd
import os
import gc
from tqdm import tqdm

def main():

    df = pd.read_csv("data/data_ger/Metadata.csv", index_col = 0)

    adatas = []
    for file in [f for f in os.listdir("data/data_ger") if f.endswith(".h5ad")]:
        print("Loading data file: ", file)

        adata = ad.read_h5ad("data/data_ger/" + file)
        adata.obs["Supertype"] = file[:-5]
        adata.obs["Donor ID"] = adata.obs["projid"]

        if "cell_type_high_resolution" not in adata.obs.columns:
            adata.obs["cell_type_high_resolution"] = adata.obs["Supertype"]

        obs_cols = {
            "projid": adata.obs["projid"], 
            "Supertype": adata.obs["Supertype"],
            "Donor ID": adata.obs["Donor ID"],
            "cell_type_high_resolution": adata.obs["cell_type_high_resolution"],
        }

        for col in tqdm(df.columns, desc="Appending columns to AnnData.obs"):
            d = dict()
            for donor_id in df["projid"].values:
                d[donor_id] = df[df["projid"] == donor_id][col].values[0]
            
            vals = [d[donor_id] for donor_id in adata.obs["projid"]]
            obs_cols[col] = vals

        adata.obs = pd.DataFrame(obs_cols)
        adata.obs["Label"] = adata.obs["cogdx"].map({1: "CT", 2: "CT", 3: "CT", 4: "AD", 5: "AD"})
        adatas.append(adata)

        print(adata)

    print("Concatenating AnnData objects")
    adata_cat = ad.concat(adatas, join="outer", label="source_file")
    print("Done")

    # Clear up some memory
    for adata in adatas:
        del adata
    del adatas
    gc.collect()

    print("Writing AnnData object to file")
    adata_cat.write_h5ad("data/data_ger/combined.h5ad")
    print("Done")

    print("Compressing AnnData object, and writing to file")
    adata_cat.write_h5ad("data/data_ger/combined_compressed.h5ad", compression="gzip")
    print("Done")


if __name__ == "__main__":
    main()