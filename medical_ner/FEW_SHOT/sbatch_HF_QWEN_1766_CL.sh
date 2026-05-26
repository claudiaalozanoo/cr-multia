#!/usr/bin/bash

#SBATCH --job-name=qwen_1766notes  
#SBATCH --output=/ijc/LABS/SOLE/DATA/tfm_CLG/medical_ner/HF/logs/qwen_1766_%j.log 
#SBATCH --mail-type=END,FAIL           
#SBATCH --mail-user=clozano@carrerasresearch.org
  
#SBATCH --partition=test-gpu            
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=4       
#SBATCH --mem=50gb 
#SBATCH --time=5:00:00 

# load modules and environment

module load Miniconda3 
source activate /mnt/beegfs/clozano/.conda/envs/medical_ner_env
conda list
 
python ner_HF_QWEN_1766_CL.py 

