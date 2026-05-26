#!/usr/bin/bash

#SBATCH --job-name=gemma_520notes  
#SBATCH --output=/ijc/LABS/SOLE/DATA/tfm_CLG/attribute_association/PLAIN/logs/gemma_520_%j.log 
#SBATCH --mail-type=END,FAIL           
#SBATCH --mail-user=clozano@carrerasresearch.org
  
#SBATCH --partition=test-gpu            
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=4       
#SBATCH --mem=100gb 
#SBATCH --time=4:00:00 

# load modules and environment

module load Miniconda3 
source activate /mnt/beegfs/clozano/.conda/envs/medical_ner_env
conda list
 
python attr_GEMMA_520_CL.py 

