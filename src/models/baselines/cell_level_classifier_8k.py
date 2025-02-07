import anndata as ad
import numpy as np
import scipy.sparse as sp

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

# FILE_PATH = "data/adata_rosmap_v3_top1000_s3_k30_drop_nout.h5ad"  # Rosmap 1k
# FILE_PATH = "data/adata_rosmap_v3_top2000_k30_drop.h5ad"  # Rosmap 2k
# FILE_PATH = "data/adata_rosmap_v3_top5000_k30_drop.h5ad"  # Rosmap 5k
FILE_PATH = "data/adata_rosmap_v3_top8000_k30_drop.h5ad"  # Rosmap 8k
# FILE_PATH = ""  # ...

N_RUNS = 10
N_FOLDS = 5
SPLIT_SEED = None


adata_sc = ad.read_h5ad(FILE_PATH)

OUTLIERS = ['11326252', '11624423', '15114174', '15144878', '20147440', '20225925', '20730959', '50101785', '50105725', '50107583']
OUTLIERS = [int(x) for x in OUTLIERS]

DONORS = sorted(adata_sc.obs["Donor ID"].unique())
DONORS = [x for x in DONORS if x not in OUTLIERS]
new_donors = []
donor_labels = []

N_CELLS = adata_sc[adata_sc.obs["Donor ID"].isin(DONORS)].shape[0]

x = sp.lil_matrix((N_CELLS, adata_sc.shape[1]))
y = np.zeros(N_CELLS, dtype=int)
donor_ids = []
start_idx = 0

for i, donor in enumerate(tqdm(DONORS)):
    adata_donor = adata_sc[adata_sc.obs["Donor ID"] == donor]

    # Get Wang labels
    cogdx = adata_sc.obs["cogdx"].loc[adata_donor.obs_names[0]]
    braaksc = adata_sc.obs["braaksc"].loc[adata_donor.obs_names[0]]
    ceradsc = adata_sc.obs["ceradsc"].loc[adata_donor.obs_names[0]]

    if cogdx == 1 and braaksc <= 3 and ceradsc >= 3:
        label = "CT"
    elif cogdx == 4 and braaksc >= 4 and ceradsc <= 2:
        label = "AD"
    else:
        label = "Other"

    x[start_idx:start_idx + adata_donor.shape[0]] = adata_donor.X
    if label == "AD":
        y[start_idx:start_idx + adata_donor.shape[0]] = 1
    
    if label in ["CT", "AD"]:
        start_idx += adata_donor.shape[0]
        new_donors.append(donor)
        donor_labels.append(label)
        donor_ids.append([donor] * adata_donor.shape[0])
    # Otherwise, skip this donor

x = x[:start_idx]  # Remove unused rows from the end
y = y[:start_idx]

x = x.tocsr()  # Convert to CSR format, for faster slicing
donor_ids = np.concatenate(donor_ids)


mean_acc_cell = np.zeros(N_RUNS)
mean_prec_cell = np.zeros(N_RUNS)
mean_rec_cell = np.zeros(N_RUNS)
mean_f1_cell = np.zeros(N_RUNS)
mean_auc_cell = np.zeros(N_RUNS)

mean_acc_donor = np.zeros(N_RUNS)
mean_prec_donor = np.zeros(N_RUNS)
mean_rec_donor = np.zeros(N_RUNS)
mean_f1_donor = np.zeros(N_RUNS)
mean_auc_donor = np.zeros(N_RUNS)

y_pred = []

for j in range(N_RUNS):
    print(f"Run {j+1}/{N_RUNS}")

    if SPLIT_SEED is not None:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    else:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)

    test_acc_cell = []
    test_acc_donor = []
    test_prec_cell = []
    test_prec_donor = []
    test_rec_cell = []
    test_rec_donor = []
    test_f1_cell = []
    test_f1_donor = []
    test_auc_cell = []
    test_auc_donor = []
    
    for i, (train_index, test_index) in enumerate(kf.split(new_donors, donor_labels)):
        # print(f"Fold {i+1}/{N_FOLDS}")

        # Train/test split
        train_donors = [new_donors[donor_idx] for donor_idx in train_index]
        test_donors = [new_donors[donor_idx] for donor_idx in test_index]
        x_train = x[np.isin(donor_ids, train_donors)]
        y_train = y[np.isin(donor_ids, train_donors)]    
        x_test = x[np.isin(donor_ids, test_donors)]
        y_test = y[np.isin(donor_ids, test_donors)]
        
        # Model definition
        model = Lasso(alpha=0.03)

        # Model training
        model.fit(x_train, y_train)

        # Model evaluation
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        test_auc_cell.append(roc_auc_score(y_test, y_pred_test))
        
        y_pred_train = (y_pred_train > 0.5).astype(int)
        y_pred_test = (y_pred_test > 0.5).astype(int)
                
        # Metrics per cell
        acc_train = (y_pred_train == y_train).mean()
        acc_test = (y_pred_test == y_test).mean()
    
        test_acc_cell.append(accuracy_score(y_test, y_pred_test))
        test_prec_cell.append(precision_score(y_test, y_pred_test))
        test_rec_cell.append(recall_score(y_test, y_pred_test))
        test_f1_cell.append(f1_score(y_test, y_pred_test))
    
        # But now we want to aggregate predictions per donor
        donor_pred = []
        donor_true = []
        test_donor_ids = np.array(donor_ids)[np.isin(donor_ids, test_donors)]
        for donor in test_donors:
            idx = np.isin(test_donor_ids, donor)
            donor_pred.append(y_pred_test[idx].mean())
            donor_true.append(y_test[idx].mean())
        donor_pred = np.array(donor_pred)
        donor_true = np.array(donor_true)

        # Also for training data
        donor_pred_train = []
        donor_true_train = []
        train_donor_ids = np.array(donor_ids)[np.isin(donor_ids, train_donors)]
        for donor in train_donors:
            idx = np.isin(train_donor_ids, donor)
            donor_pred_train.append(y_pred_train[idx].mean())
            donor_true_train.append(y_train[idx].mean())
        donor_pred_train = np.array(donor_pred_train)
        donor_true_train = np.array(donor_true_train)

        # Pick best threshold based on training data:
        thresholds = np.linspace(0, 1, 101)
        best_threshold = 0
        best_acc = 0
        for threshold in thresholds:
            donor_pred_train_ = (donor_pred_train > threshold).astype(int)
            acc = accuracy_score(donor_true_train, donor_pred_train_)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold


        # Metrics per donor
        test_auc_donor.append(roc_auc_score(donor_true, donor_pred))
        donor_pred = (donor_pred > best_threshold).astype(int)

        test_acc_donor.append(accuracy_score(donor_true, donor_pred))
        test_prec_donor.append(precision_score(donor_true, donor_pred))
        test_rec_donor.append(recall_score(donor_true, donor_pred))
        test_f1_donor.append(f1_score(donor_true, donor_pred))

        # print("With best threshold (gridsearch):")
        # print(f"acc  (cell) = {test_acc_cell[-1]:.4f}, acc  (donor) = {test_acc_donor[-1]:.4f}")
        # print(f"prec (cell) = {test_prec_cell[-1]:.4f}, prec (donor) = {test_prec_donor[-1]:.4f}")
        # print(f"rec  (cell) = {test_rec_cell[-1]:.4f}, rec  (donor) = {test_rec_donor[-1]:.4f}")
        # print(f"f1   (cell) = {test_f1_cell[-1]:.4f}, f1   (donor) = {test_f1_donor[-1]:.4f}")
        # print(f"auc  (cell) = {test_auc_cell[-1]:.4f}, auc  (donor) = {test_auc_donor[-1]:.4f}")


    mean_acc_cell[j] = np.mean(test_acc_cell)
    mean_prec_cell[j] = np.mean(test_prec_cell)
    mean_rec_cell[j] = np.mean(test_rec_cell)
    mean_f1_cell[j] = np.mean(test_f1_cell)
    mean_auc_cell[j] = np.mean(test_auc_cell)

    mean_acc_donor[j] = np.mean(test_acc_donor)
    mean_prec_donor[j] = np.mean(test_prec_donor)
    mean_rec_donor[j] = np.mean(test_rec_donor)
    mean_f1_donor[j] = np.mean(test_f1_donor)
    mean_auc_donor[j] = np.mean(test_auc_donor)
    
print(f"Mean test  acc (cell): {np.mean(mean_acc_cell):.4f} +/- {np.std(mean_acc_cell):.4f}")
print(f"Mean test prec (cell): {np.mean(mean_prec_cell):.4f} +/- {np.std(mean_prec_cell):.4f}")
print(f"Mean test  rec (cell): {np.mean(mean_rec_cell):.4f} +/- {np.std(mean_rec_cell):.4f}")
print(f"Mean test   f1 (cell): {np.mean(mean_f1_cell):.4f} +/- {np.std(mean_f1_cell):.4f}")
print(f"Mean test  auc (cell): {np.mean(mean_auc_cell):.4f} +/- {np.std(mean_auc_cell):.4f}")

print("\nPer donor:")
print(f"Mean test  acc: {np.mean(mean_acc_donor):.4f} +/- {np.std(mean_acc_donor):.4f}")
print(f"Mean test prec: {np.mean(mean_prec_donor):.4f} +/- {np.std(mean_prec_donor):.4f}")
print(f"Mean test  rec: {np.mean(mean_rec_donor):.4f} +/- {np.std(mean_rec_donor):.4f}")
print(f"Mean test   f1: {np.mean(mean_f1_donor):.4f} +/- {np.std(mean_f1_donor):.4f}")
print(f"Mean test  auc: {np.mean(mean_auc_donor):.4f} +/- {np.std(mean_auc_donor):.4f}")