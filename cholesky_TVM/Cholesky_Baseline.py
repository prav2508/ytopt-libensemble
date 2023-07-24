import numpy as np
import tvm
from tvm import autotvm

import time
from tvm import te
import logging
import argparse
from datetime import datetime
import json

logData = {}

logPath = 'logs/'
resultsPath = 'results/'

def cholesky_basic(N, M, dtype):

    A = te.placeholder((N, M), name="A", dtype=dtype)
    At = te.placeholder((M, N), name="At", dtype=dtype)

    k = te.reduce_axis((0, M), name="k")
    C = te.compute((N, N), lambda i, j: te.sum(A[i, k] * At[k, j], axis=k), name="C")

    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    
    #using 8 as the tiling factor
    yo, yi = s[C].split(y, 8) 
    xo, xi = s[C].split(x, 8)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, At, C]




def main(datasize):
    if datasize == 'L':  
        N, M = 1000 , 1200
    else:
        N, M = 2000, 2600
    s, arg_bufs = cholesky_basic(N, M, "float64")
    func = tvm.build(s,arg_bufs)
    a_np = np.random.uniform(size=(N, M)).astype(np.float64)
    b_np = a_np.T
    c_np = a_np.dot(b_np)

    c_tvm = tvm.nd.empty(c_np.shape, dtype='float64')
    
    start = time.time()
    func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)
    end = time.time()

    runTime = round(end-start,2)
    logging.info("Elpased time = {} secs".format(runTime))
    logData['ElapsedTime'] = runTime
    with open(resultsPath+'Baseline.json', "w") as json_file:
        json.dump(logData, json_file)


if __name__ == '__main__':

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--size', type=str, choices=['L', 'XL'], help='L or XL')
        args = parser.parse_args()
        size = args.size
        if size == 'L':
            logPath = logPath + 'large/'
            resultsPath = resultsPath + 'large/'
        else:
            logPath = logPath + 'extraLarge/'
            resultsPath = resultsPath + 'extraLarge/'

        logging.basicConfig(filename=logPath+'Baseline_MM.log', level=logging.DEBUG,filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        logging.info('TVM Baseline Matrix Multiplication - {}'.format(datetime.now()))

        main(size)

    except Exception as e:
        logging.error("Exception occured = ",e)

    