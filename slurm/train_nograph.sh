#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400GB
#SBATCH --mail-type=END
#SBATCH --output=slurm/out/%j_%a_train_nograph.out
#SBATCH --error=slurm/out/%j_%a_train_nograph.out
#SBATCH --gres=gpu
#SBATCH --array=0-8
/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

module use /opt/insy/modulefiles
module load cuda/12.1

DATASETS=(
	rosmap_mit_hvg1000_k30.h5ad
	rosmap_mit_hvg1000_postnorm_k30.h5ad
	rosmap_mit_hvg2000_k30.h5ad
	rosmap_mit_hvg2000_postnorm_k30.h5ad
	rosmap_mit_hvg3000_k30.h5ad
	rosmap_mit_hvg3000_postnorm_k30.h5ad
	rosmap_mit_lieke_allgenes_hvg2k_k30.h5ad
	rosmap_mit_lieke_allgenes_hvg2k_postnorm_k30.h5ad
	rosmap_mit_lieke_hvg_k30.h5ad
)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
NAME=${DATASET%.h5ad}

OUTDIR=out/results/lieke
mkdir -p ${OUTDIR}

echo "=== Running dataset: ${DATASET} (array task ${SLURM_ARRAY_TASK_ID}) ==="

# Step 1: 5-fold CV for performance evaluation
echo "--- Starting 5-fold CV ---"
apptainer exec --nv --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind data/:/opt/app/data/ \
	--bind out/:/opt/app/out/ \
	scAGG_container.sif pixi run python src/train.py \
		--dataset data/lieke/${DATASET} \
		--metadata data/rosmap_meta/ROSMAP_clinical.csv \
		--meta-sample-col projid \
		--n-epochs 2 \
		--dim 32 \
		--split-seed 42 \
		--batch-size 8 \
		--dropout 0.1 \
		--learning-rate 0.001 \
		--pooling mean \
		--label wang \
		--n_splits 5 \
		--no-graph \
		--verbose \
		--save \
		--output ${OUTDIR}/${NAME}_cv.h5ad

# Step 2: Train on all data
echo "--- Starting full training ---"
apptainer exec --nv --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind data/:/opt/app/data/ \
	--bind out/:/opt/app/out/ \
	scAGG_container.sif pixi run python src/train_full.py \
		--dataset data/lieke/${DATASET} \
		--metadata data/rosmap_meta/ROSMAP_clinical.csv \
		--meta-sample-col projid \
		--n-epochs 2 \
		--dim 32 \
		--split-seed 42 \
		--batch-size 8 \
		--dropout 0.1 \
		--learning-rate 0.001 \
		--pooling mean \
		--label wang \
		--no-graph \
		--verbose \
		--save \
		--save-embeddings \
		--output ${OUTDIR}/${NAME}_full.h5ad \
		--output-model ${OUTDIR}/${NAME}_full.pt
