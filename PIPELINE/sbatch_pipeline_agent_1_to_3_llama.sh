#!/usr/bin/bash

#SBATCH --job-name=pipeline_llama_agents1_3
#SBATCH --output=/ijc/LABS/SOLE/DATA/tfm_CLG/PIPELINE/logs/pipeline_llama_1_3_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=clozano@carrerasresearch.org

#SBATCH --partition=test-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100gb
#SBATCH --time=1:00:00

# load modules and environment

module load Miniconda3
source activate /mnt/beegfs/clozano/.conda/envs/medical_ner_env
conda list

python pipeline_agent_1_to_3_llama.py "$1"

