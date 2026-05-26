#!/usr/bin/bash

#SBATCH --job-name=wes  
#SBATCH --output=%nextflow_sarek%j.log 
#SBATCH --mail-type=END,FAIL           
#SBATCH --mail-user=emancini@carrerasresearch.org  
#SBATCH --cpus-per-task=2 # Run on a single CPU
#SBATCH --mem=24gb 
#SBATCH --time=100:00:00 

# Cargar mos necesarios
module load Nextflow/23.04.3
module load singularity/3.11.4-GCC-11.2.0

 
nextflow run /ijc/LABS/SOLE/DATA/nf-core/sarek  -profile slurm,singularity --outdir p01_wes -w ~/wd  --genome GATK.GRCh38  -c p01_wes.config 
 
 

