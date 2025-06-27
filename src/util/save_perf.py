import json
import os


# structure of the json:
# {
#     "model": [  # list over the folds
#         {
#             "fold": 0,  # matches index in the list
#             "accuracy: float,
#             "precision": float,
#             ...
#             "train_donors": [ ... ],
#             "test_donors": [ ... ],
#             "train_y": [ ... ],  # label for each donor in train_donors
#             "test_y": [ ... ],  # label for each donor in test_donors
#             "train_y_pred": [ ... ],  # predicted label for each donor in train_donors
#             "test_y_pred": [ ... ],  # predicted label for each donor in test_donors
#         },
#     ],
# }


def save_perf(
        exp_name: str,  # determines the file name
        model_name: str,  # determines the model name in the json file
        fold: int,  # the index of the fold, used to match the model in the list

        accuracy: float = None,  # If None, test_y and test_y_pred must be provided, so we can calculate it
        precision: float = None,
        recall: float = None,
        f1: float = None,
        roc_auc: float = None,
        
        train_donors: list = None,
        test_donors: list = None,
        train_y: list = None,
        test_y: list = None,
        train_y_pred: list = None,
        test_y_pred: list = None,

        **kwargs,  # any other keyworded arguments, these will be saved as extra keys
):
    
    BASE_PATH = "out/results/"
    file_path = f"{BASE_PATH}{exp_name}.json"

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    
    if model_name not in data:
        data[model_name] = []

    if type(train_donors) is not list:
        train_donors = train_donors.tolist()
    if type(test_donors) is not list:
        test_donors = test_donors.tolist()
    if type(train_y) is not list:
        train_y = train_y.tolist()
    if type(test_y) is not list:
        test_y = test_y.tolist()
    if type(train_y_pred) is not list:
        train_y_pred = train_y_pred.tolist()
    if type(test_y_pred) is not list:
        test_y_pred = test_y_pred.tolist()

    # make the fold dict
    fold_dict = {
        "fold": fold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "train_donors": train_donors,
        "test_donors": test_donors,
        "train_y": train_y,
        "test_y": test_y,
        "train_y_pred": train_y_pred,
        "test_y_pred": test_y_pred,
    }

    for key, value in kwargs.items():
        fold_dict[key] = value

    # append the fold dict to the model list

    # put empty dicts for missing folds
    while len(data[model_name]) <= fold:
        data[model_name].append({})

    data[model_name][fold] = fold_dict

    # save the data to the file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    


if __name__ == "__main__":
    # Example usage
    # check that it works with numpy arrays as well

    import numpy as np

    save_perf(
        exp_name="test_exp",
        model_name="test_model",
        fold=0,
        accuracy=0.95,
        precision=0.9,
        recall=0.85,
        f1=0.88,
        roc_auc=0.92,
        train_donors=np.array(["donor1", "donor2"]),
        test_donors=np.array(["donor3", "donor4"]),
        train_y=np.array([1, 0]),
        test_y=np.array([1, 0]),
        train_y_pred=np.array([1, 0]),
        test_y_pred=np.array([1, 0]),
        extra_metric=0.99  # example of an extra metric
    )