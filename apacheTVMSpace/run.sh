#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --job-name=ytopt
#SBATCH --output=slurmLog/out/gpu-matmul-ytopt.%j.out
#SBATCH --error=slurmLog/err/gpu-matmul-ytopt.%j.err
#SBATCH --gres=gpu:8
#SBATCH --account=EE-ECP
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pparamasivam@anl.gov

source /home/pparamasivam/anaconda3/etc/profile.d/conda.sh
cd /home/pparamasivam/ytune/ytopt-libensemble/apacheTVMSpace

conda activate ytune

python matMul_GATuner.py
python matMul_GridSearchTuner.py
python matMul_RandTuner.py
python matMul_XGBTuner.py
