:: Save this file with a .bat extension (e.g., stability_exp.bat)

:: base scAGG on ROSMAP
:: pixi run python src/train.py --dataset data/adata_rosmap_v3_top1000_s3_k30_drop_nout.h5ad --n-epochs 2 --dim 32 --split-seed 42 --batch-size 8 --dropout 0.1 --pooling mean --label wang --n_splits 5 --no-graph --output out/results/ROSMAP_scAGG_results.h5ad

:: Run 5 times with the same split seed, save to numbered output files
FOR /L %%i IN (1,1,5) DO (
    pixi run python src/train.py --dataset data/adata_rosmap_v3_top1000_s3_k30_drop_nout.h5ad --n-epochs 2 --dim 32 --split-seed 42 --batch-size 8 --dropout 0.1 --pooling mean --label wang --n_splits 5 --no-graph --output out/results/ROSMAP_scAGG_results_rep%%i.h5ad
)

:: Run 5 times with different split seeds
FOR /L %%i IN (42,1,46) DO (
    pixi run python src/train.py --dataset data/adata_rosmap_v3_top1000_s3_k30_drop_nout.h5ad --n-epochs 2 --dim 32 --split-seed %%i --batch-size 8 --dropout 0.1 --pooling mean --label wang --n_splits 5 --no-graph --output out/results/ROSMAP_scAGG_results_split%%i.h5ad
)
