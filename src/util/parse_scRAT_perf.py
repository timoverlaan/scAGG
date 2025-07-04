import re

from save_perf import save_perf


if __name__ == "__main__":

    # Example line:
    # "Best performance: Epoch 2, Loss 0.077504, Test ACC 0.964286, Test AUC 1.000000, Test Recall 1.000000, Test Precision 0.909091"
    
    with open("out/results/scRAT_COMBAT.out", "r") as f:
        fold = 0
        for line in f:
        
            # Extract ACC, AUC, Recall and Precision. Calculate f1
            match = re.search(
                r"Best performance: Epoch \d+, Loss [\d.]+, Test ACC ([\d.]+), Test AUC ([\d.]+), Test Recall ([\d.]+), Test Precision ([\d.]+)",
                line,
            )

            if match:
                acc = float(match.group(1))
                auc = float(match.group(2))
                recall = float(match.group(3))
                precision = float(match.group(4))

                # Calculate F1 score
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

                print(f"Fold: {fold}, ACC: {acc}, AUC: {auc}, Recall: {recall}, Precision: {precision}, F1: {f1}")

                save_perf(
                    exp_name="COMBAT_top2000",
                    model_name="scRAT",
                    fold=fold,
                    accuracy=acc,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    roc_auc=auc,
                )

                fold += 1

