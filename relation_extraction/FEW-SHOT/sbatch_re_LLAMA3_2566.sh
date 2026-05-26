#!/usr/bin/bash

#SBATCH --job-name=llama3_v2_agent3_2566  
#SBATCH --output=/ijc/LABS/SOLE/DATA/tfm_CLG/relation_extraction/ZERO-SHOT/logs/llama3_2566_%j.log 
#SBATCH --mail-type=END,FAIL           
#SBATCH --mail-user=clozano@carrerasresearch.org
  
#SBATCH --partition=test-gpu            
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=4       
#SBATCH --mem=150gb 
#SBATCH --time=4:00:00 

# load modules and environment

module load Miniconda3 
source activate /mnt/beegfs/clozano/.conda/envs/medical_ner_env
conda list
 
python re_LLAMA_2566.py 

