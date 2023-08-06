#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --job-name=ytopt
#SBATCH --output=slurmLog/out/gpu-matmul-tvm.%j.out
#SBATCH --error=slurmLog/err/gpu-matmul-tvm.%j.err
#SBATCH --gres=gpu:8
#SBATCH --account=STARTUP-PPARAMASIVAM
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pparamasivam@anl.gov

source /home/pparamasivam/anaconda3/bin/activate
conda activate tvm

python /home/pparamasivam/ytune/ytopt-libensemble/cholesky_TVM/Cholesky_Baseline.py --size=L
python /home/pparamasivam/ytune/ytopt-libensemble/cholesky_TVM/Cholesky_GATuner.py --size=L
python /home/pparamasivam/ytune/ytopt-libensemble/cholesky_TVM/Cholesky_GridSearch.py --size=L
python /home/pparamasivam/ytune/ytopt-libensemble/cholesky_TVM/Cholesky_RandTuner.py --size=L
python /home/pparamasivam/ytune/ytopt-libensemble/cholesky_TVM/Cholesky_XGBTuner.py --size=L
