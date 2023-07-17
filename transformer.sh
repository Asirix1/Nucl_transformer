#!/bin/sh
#SBATCH --job-name=transformer          
#SBATCH --error=transformer-%j.err        
#SBATCH --output=transformer-%j.log        
#SBATCH --time=71:00:00  
#SBATCH --partition=operation
#SBATCH --nodes 1                  
#SBATCH --cpus-per-task=96             
#SBATCH --mem=1000G
source /home/popov/miniconda3/bin/activate
conda activate /home/popov/miniconda3/envs/apopov
for i in intergenic_regions exons introns 5UTR 3UTR 
do 
export i 
srun python3.8 /beegfs/data/hpcws/ws1/popov-transformer_work/Nucleotide_trans.py
done
srun python3.8 /beegfs/data/hpcws/ws1/popov-transformer_work/tSNE.py
              