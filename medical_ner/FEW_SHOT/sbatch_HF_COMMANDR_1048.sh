#!/usr/bin/bash

#SBATCH --job-name=commandr_1048notes  
#SBATCH --output=/ijc/LABS/SOLE/DATA/tfm_CLG/medical_ner/HF/logs/commandr_1048_%j.log 
#SBATCH --mail-type=END,FAIL           
#SBATCH --mail-user=clozano@carrerasresearch.org
  
#SBATCH --partition=test-gpu            
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=4       
#SBATCH --mem=100gb 
#SBATCH --time=6:00:00 

# load modules and environment

module load Miniconda3 
source activate /mnt/beegfs/clozano/.conda/envs/medical_ner_env
conda list

export SAFETENSORS_FAST_GPU=0

ulimit -v unlimited
 
python ner_HF_COMMANDR_1048_CL.py 

