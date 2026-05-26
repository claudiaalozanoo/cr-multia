#!/usr/bin/bash

#SBATCH --job-name=llama3_2566notes  
#SBATCH --output=/ijc/LABS/SOLE/DATA/tfm_CLG/attribute_association/PLAIN/logs/llama3_2566_%j.log 
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
 
python attr_LLAMA3_2566_CL.py 

