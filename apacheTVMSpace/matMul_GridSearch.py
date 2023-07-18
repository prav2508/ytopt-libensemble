import numpy as np
import tvm
from tvm import autotvm
import logging
import sys
import time
from tvm import te
import logging
from datetime import datetime
import argparse

logPath = 'logs/'
resultsPath = 'results/'


def record_execution_time(task, config, duration):
    execution_time = duration
    logging.info(f"Execution time: {execution_time} seconds")


@autotvm.template("test/tvmmatmul_v1") 
def matmul_v1(N, L, M, size, dtype):

    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # 2. get the config object
    cfg = autotvm.get_config()

    # 3. define search space 

    if size != 'L':

        cfg.define_knob("tile_y", [1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000, 2000])
        cfg.define_knob("tile_x", [1, 2, 4, 5, 10, 13, 20, 25, 40, 50, 52, 100, 104, 200, 260, 325, 520, 650, 1300, 2600])
    else:    

        cfg.define_knob("tile_y", [1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000])
        cfg.define_knob("tile_x", [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 40, 48, 50, 60, 75, 80, 100, 120, 150, 200, 240, 300, 400, 600, 1200]) 

    # 4. schedule according to config
    yo, yi = s[C].split(y, cfg["tile_y"].val)
    xo, xi = s[C].split(x, cfg["tile_x"].val)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


def main(datasize):

    if datasize == 'L':  
        N, L, M = 1000 , 1100 , 1200
    else:
        N, L, M = 2000, 2300, 2600

    task = autotvm.task.create("test/tvmmatmul_v1", args=(N, L, M, datasize ,"float64"), target="llvm")


    # Create a measurement callback with the custom function

    measure_option = autotvm.measure_option(builder="local", 
                                            runner=autotvm.LocalRunner(number=1, repeat=1, timeout=200), # timeout=20
                                            )

    tuner = autotvm.tuner.GridSearchTuner(task)
    path = resultsPath + "tvmGridSearchTuner.json"
    start = time.time()
    tuner.tune(
    n_trial=100,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file(path)]
    )
    end = time.time()

    logging.info("Elpased time = {}".format(end-start))

    with autotvm.apply_history_best(path):
        with tvm.target.Target("llvm"):
            s, arg_bufs = matmul_v1(N, L, M,datasize,"float64") #float64 
            func = tvm.build(s, arg_bufs)

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

        logging.basicConfig(filename=logPath+'GridSearchTuner.log', level=logging.DEBUG,filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        logging.info('GridSearch Tuner Matrix Multiplication - {}'.format(datetime.now()))


        main(size)

    except Exception as e:
        logging.error("Exception occured = ",e)

