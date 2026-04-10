#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --mail-type=END
#SBATCH --output=slurm/out/%j_%a_eval_seaad.out
#SBATCH --error=slurm/out/%j_%a_eval_seaad.out
#SBATCH --gres=gpu
#SBATCH --array=0-2
/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

module use /opt/insy/modulefiles
module load cuda/12.1

# Pairs: trained model name (under out/results/lieke/<name>_full.pt)
# matched with the SeaAD validation h5ad in data/seaad/.
TRAINED=(
	rosmap_mit_lieke_allgenes_hvg2k_k30
	rosmap_mit_lieke_allgenes_hvg2k_postnorm_k30
	rosmap_mit_lieke_hvg_k30
)
SEAAD=(
	seaad_lieke_allgenes_hvg2k_k30.h5ad
	seaad_lieke_allgenes_hvg2k_postnorm_k30.h5ad
	seaad_lieke_hvg_k30.h5ad
)

NAME=${TRAINED[$SLURM_ARRAY_TASK_ID]}
DATASET=${SEAAD[$SLURM_ARRAY_TASK_ID]}

MODEL_DIR=out/results/lieke
OUTDIR=out/results/lieke_seaad
mkdir -p ${OUTDIR}

echo "=== Evaluating ${NAME} on ${DATASET} (array task ${SLURM_ARRAY_TASK_ID}) ==="

apptainer exec --nv --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind data/:/opt/app/data/ \
	--bind out/:/opt/app/out/ \
	scAGG_container.sif pixi run python src/eval.py \
		--dataset data/lieke/seaad/${DATASET} \
		--model ${MODEL_DIR}/${NAME}_full.pt \
		--label-col Wang \
		--intermediate-col Wang_intermediate \
		--positive-label AD \
		--negative-label Healthy \
		--compute-wang \
		--save-embeddings \
		--verbose \
		--output ${OUTDIR}/${NAME}_seaad.h5ad \
		--output-metrics ${OUTDIR}/${NAME}_seaad_metrics.json
