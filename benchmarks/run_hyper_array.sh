#!/bin/bash
#SBATCH --job-name=infer_hyper
#SBATCH --output=logs/slurm_infer_hyper_%A_%a.out
#SBATCH --error=logs/slurm_infer_hyper_%A_%a.err
#SBATCH --array=1-144
#SBATCH --gpus=1
#SBATCH --time=02:00:00

module load cuda/12.6
source ~/.bashrc
mamba activate gpu

cd /home/s5f/twarf.s5f/maces/mace

PARAM_FILE="benchmarks/inference_hypers_param_grid.csv"
LINE=$(($SLURM_ARRAY_TASK_ID + 1))  # skip header
PARAMS=$(sed -n "${LINE}p" "$PARAM_FILE")

IFS=',' read -r max_ell hidden_irreps edge_irreps <<< "$PARAMS"


label="${edge_irreps//[$'\n\r']/}-${hidden_irreps//[$'\n\r']/}-${max_ell//[$'\n\r']/}"
echo "${label}"


# Run pytest for this parameter set
pytest tests/test_inference_hypers.py \
    -v \
    -k "${label}" \
    --benchmark-save="infer_hyper_${SLURM_ARRAY_TASK_ID}"
