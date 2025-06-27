

exit 0  # Don't run this script directly, it's just for reference.

# This script contains commands to run various experiments with scAGG.

######### ROSMAP #########

# base scAGG on ROSMAP
# pixi run python src/train.py --dataset data/adata_rosmap_v3_top1000_s3_k30_drop_nout.h5ad --n-epochs 2 --dim 32 --split-seed 42 --batch-size 8 --dropout 0.1 --pooling mean --label wang --n_splits 5 --no-graph --output out/results/ROSMAP_scAGG_results.h5ad
# Mean performance over all folds:
#         accuracy over all folds: 0.7198 (+-0.0678)
#        precision over all folds: 0.7267 (+-0.0784)
#           recall over all folds: 0.8200 (+-0.0281)
#               f1 over all folds: 0.7682 (+-0.0478)
#          roc_auc over all folds: 0.7860 (+-0.1005)

# scAGG + GAT
# pixi run python src/train.py --dataset data/adata_rosmap_v3_top1000_s3_k30_drop_nout.h5ad --dim 16 --split-seed 42 --batch-size 8 --dropout 0.5 --pooling mean --label wang --n_splits 5  --n-epochs 5 --save --output out/results/ROSMAP_scAGG+GAT_results.h5ad
# Mean performance over all folds:
#         accuracy over all folds: 0.7056 (+-0.0550)
#        precision over all folds: 0.6989 (+-0.0558)
#           recall over all folds: 0.8450 (+-0.0685)
#               f1 over all folds: 0.7628 (+-0.0451)
#          roc_auc over all folds: 0.7611 (+-0.1013)



# with the Mathys DEGs
pixi run python src/train.py --dataset data/rosmap_mit_Mathysgenes_k30.h5ad --n-epochs 2 --dim 32 --split-seed 42 --batch-size 8 --dropout 0.1 --pooling mean --label wang --n_splits 5 --no-graph
# Mean performance over all folds: (but this is on the newly processed ROSMAP, so it still has the outliers, should compare with: w:\staff-umbrella\scGraphNN\rosmap-processing\data\processed\rosmap_mit_top1000_k30.h5ad)
#     accuracy over all folds: 0.6882 (+-0.0711)
#    precision over all folds: 0.7193 (+-0.0493)
#       recall over all folds: 0.8437 (+-0.0746)
#           f1 over all folds: 0.7755 (+-0.0537)
#      roc_auc over all folds: 0.7430 (+-0.0341)

# with new processed rosmap top1000 (Mathys genes performance should be compared with this)
pixi run python src/train.py --dataset data/rosmap_mit_top1000_k30.h5ad --n-epochs 2 --dim 32 --split-seed 42 --batch-size 8 --dropout 0.1 --pooling mean --label wang --n_splits 5 --no-graph
# Mean performance over all folds:
#         accuracy over all folds: 0.7000 (+-0.0570)
#        precision over all folds: 0.7331 (+-0.0535)
#           recall over all folds: 0.8437 (+-0.0378)
#               f1 over all folds: 0.7835 (+-0.0378)
#          roc_auc over all folds: 0.7331 (+-0.0329)
# (so the improvement of using DEGs is marginal)


######### COMBAT #########

# base scAGG on COMBAT
pixi run python src/train.py --dataset data/COMBAT/COMBAT-CITESeq-DATA-top2000.h5ad --n-epochs 2 --dim 32 --split-seed 42 --batch-size 8 --dropout 0.1 --pooling mean --label SARSCoV2PCR --n_splits 5 --no-graph --output out/results/COMBAT_top2000_scAGG_results.h5ad

# with GAT
pixi run python src/train.py --dataset data/COMBAT/COMBAT-CITESeq-DATA-top2000_k30.h5ad --dim 16 --split-seed 42 --batch-size 8 --dropout 0.5 --pooling mean --label SARSCoV2PCR --n_splits 5  --n-epochs 5 --save --output out/results/COMBAT_top2000_scAGG+GAT_results.h5ad

# with attention pooling (AP)
pixi run python src/train.py --dataset data/COMBAT/COMBAT-CITESeq-DATA-top2000.h5ad --n-epochs 2 --dim 32 --split-seed 42 --batch-size 8 --dropout 0.1 --pooling self-att-sum --label SARSCoV2PCR --n_splits 5 --no-graph --output out/results/COMBAT_top2000_scAGG+AP_results.h5ad

# with GAT + AP
pixi run python src/train.py --dataset data/COMBAT/COMBAT-CITESeq-DATA-top2000_k30.h5ad --dim 16 --split-seed 42 --batch-size 8 --dropout 0.5 --pooling self-att-sum --label SARSCoV2PCR --n_splits 5  --n-epochs 5 --save --output out/results/COMBAT_top2000_scAGG+GAT+AP_results.h5ad
