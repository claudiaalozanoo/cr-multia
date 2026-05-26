#!/usr/bin/bash

#SBATCH --job-name=bert_finetunning_agent1  
#SBATCH --output=/ijc/LABS/SOLE/DATA/tfm_CLG/medical_ner/FINETUNNING/RESULTS_BERT_V1/logs/bert_finetunning_%j.log 
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
 
python fine_tunning_BERT.py 

