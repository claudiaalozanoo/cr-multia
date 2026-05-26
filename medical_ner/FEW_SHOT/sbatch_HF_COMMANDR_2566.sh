#!/usr/bin/bash

#SBATCH --job-name=commandr_2566notes  
#SBATCH --output=/ijc/LABS/SOLE/DATA/tfm_CLG/medical_ner/HF/logs/commandr_2566_%j.log 
#SBATCH --mail-type=END,FAIL           
#SBATCH --mail-user=clozano@carrerasresearch.org
  
#SBATCH --partition=test-gpu            
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=4       
#SBATCH --mem=100gb 
#SBATCH --time=10:00:00 

# load modules and environment

module load Miniconda3 
source activate /mnt/beegfs/clozano/.conda/envs/medical_ner_env
conda list

# 1. Desactiva el uso de mmap a nivel global para la librería safetensors
export SAFETENSORS_FAST_GPU=0

# 2. Asegura que los límites de memoria virtual sean amplios
ulimit -v unlimited
 
python ner_HF_COMMANDR_2566_CL.py 

