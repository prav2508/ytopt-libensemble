#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --job-name=ytopt
#SBATCH --output=slurmLog/out/gpu-matmul-tvm.%j.out
#SBATCH --error=slurmLog/err/gpu-matmul-tvm.%j.err
#SBATCH --gres=gpu:8
#SBATCH --account=EE-ECP
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pparamasivam@anl.gov

source /home/pparamasivam/anaconda3/bin/activate
conda activate tvm
python /home/pparamasivam/ytune/ytopt-libensemble/apacheTVMSpace/matMul_GATuner.py
python /home/pparamasivam/ytune/ytopt-libensemble/apacheTVMSpace/matMul_GridSearch.py
python /home/pparamasivam/ytune/ytopt-libensemble/apacheTVMSpace/matMul_RandTuner.py
