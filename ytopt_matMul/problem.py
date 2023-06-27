import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
from autotune.problem import TuningProblem
from autotune.space import *
import os
import sys
import time
import json
import math

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from skopt.space import Real, Integer, Categorical

from plopper import Plopper
nparams = 1

cs = CS.ConfigurationSpace(seed=1234)
#tile_size
p0= CSH.OrdinalHyperparameter(name='p0', sequence=[2,4,8,16,32,64,128,256,512], default_value=8)
p1= CSH.OrdinalHyperparameter(name='p1', sequence=[2,4,8,16,32,64,128,256,512], default_value=8)

cs.add_hyperparameters([p0,p1])

# problem space
task_space = None

input_space = cs

output_space = Space([
     Real(0.0, inf, name="time")
])

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/matMulTemplate.py',dir_path)

x1=['p0','p1']

def myobj(point: dict):

  def plopper_func(x):
    print(x)
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    value = [point[x1[0]],point[x1[1]]]
    print('VALUES:',point[x1[0]])
    params = ["P0","P1"]


    result = obj.findRuntime(value, params)
    return result

  x = np.array([point[f'p{i}'] for i in range(len(point))])
  results = plopper_func(x)
  print('OUTPUT: ',results)

  return results

Problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None
    )