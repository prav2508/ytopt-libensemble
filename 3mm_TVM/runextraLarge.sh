#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=ytopt
#SBATCH --output=slurmLog/out/gpu-matmul-tvm.%j.out
#SBATCH --error=slurmLog/err/gpu-matmul-tvm.%j.err
#SBATCH --gres=gpu:8
#SBATCH --account=EE-ECP
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pparamasivam@anl.gov

source /home/pparamasivam/anaconda3/bin/activate
conda activate tvm

python /home/pparamasivam/ytune/ytopt-libensemble/tvm3mm/tvm3MM_Baseline.py --size=XL
python /home/pparamasivam/ytune/ytopt-libensemble/tvm3mm/tvm3MM_GATuner.py --size=XL
python /home/pparamasivam/ytune/ytopt-libensemble/tvm3mm/tvm3MM_GridSearchTuner.py --size=XL
python /home/pparamasivam/ytune/ytopt-libensemble/tvm3mm/tvm3MM_RandTuner.py --size=XL
python /home/pparamasivam/ytune/ytopt-libensemble/tvm3mm/tvm3MM_XGBTuner.py --size=XL