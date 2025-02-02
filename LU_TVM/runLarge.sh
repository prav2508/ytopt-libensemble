#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --job-name=ytopt
#SBATCH --output=slurmLog/out/gpu-matmul-tvm.%j.out
#SBATCH --error=slurmLog/err/gpu-matmul-tvm.%j.err
#SBATCH --gres=gpu:1
#SBATCH --account=EE-ECP
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pparamasivam@anl.gov

source /home/pparamasivam/anaconda3/bin/activate
conda activate tvm

python /home/pparamasivam/ytune/ytopt-libensemble/LU_TVM/LU_Baseline.py --size=L
python /home/pparamasivam/ytune/ytopt-libensemble/LU_TVM/LU_GATuner.py --size=L
python /home/pparamasivam/ytune/ytopt-libensemble/LU_TVM/LU_GridSearch.py --size=L
python /home/pparamasivam/ytune/ytopt-libensemble/LU_TVM/LU_RandTuner.py --size=L
python /home/pparamasivam/ytune/ytopt-libensemble/LU_TVM/LU_XGBTuner.py --size=L
