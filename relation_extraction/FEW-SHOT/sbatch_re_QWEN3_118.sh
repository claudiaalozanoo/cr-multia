#!/usr/bin/bash

#SBATCH --job-name=qwen3_v1_agent3  
#SBATCH --output=/ijc/LABS/SOLE/DATA/tfm_CLG/relation_extraction/ZERO-SHOT/logs/qwen3_v1_118_%j.log 
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
 
python re_QWEN3_118.py 

