import numpy as np
import tvm
from tvm import autotvm
import logging
import sys
import time
from tvm import te
import logging
from datetime import datetime

logging.basicConfig(filename='logs/GATuner.log', level=logging.DEBUG,filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.info('GA Tuner Matrix Multiplication - {}'.format(datetime.now()))

def record_execution_time(task, config, duration):
    execution_time = duration
    logging.info(f"Execution time: {execution_time} seconds")


@autotvm.template("test/tvmmatmul_v1") 
def matmul_v1(N, L, M, dtype):
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
    cfg.define_knob("tile_y", [2,4,8,16,32,64,128,256,512,1024])
    cfg.define_knob("tile_x", [2,4,8,16,32,64,128,256,512,1024])


    # 4. schedule according to config
    yo, yi = s[C].split(y, cfg["tile_y"].val)
    xo, xi = s[C].split(x, cfg["tile_x"].val)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


def main():

    N, L, M = 2048 , 2048 , 2048
    task = autotvm.task.create("test/tvmmatmul_v1", args=(N, L, M,"float64"), target="llvm")


    # Create a measurement callback with the custom function

    measure_option = autotvm.measure_option(builder="local", 
                                            runner=autotvm.LocalRunner(number=1, repeat=1, timeout=200), # timeout=20
                                            )

    tuner = autotvm.tuner.GATuner(task)
    start = time.time()
    tuner.tune(
    n_trial=100,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("results/tvm_GATuner.json")]
    )
    end = time.time()

    logging.info("Elpased time = {}".format(end-start))

    with autotvm.apply_history_best("results/tvm_GATuner.json"):
        with tvm.target.Target("llvm"):
            s, arg_bufs = matmul_v1(N, L, M,"float64") #float64 
            func = tvm.build(s, arg_bufs)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error("Exception occured = ",e)

