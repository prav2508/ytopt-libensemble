import numpy as np
import tvm
from tvm import autotvm
import logging
import sys
import time
from tvm import te

def record_execution_time(task, config, duration):
    execution_time = duration
    print(f"Execution time: {execution_time} seconds")


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
    cfg.define_knob("tile_y", [2,4,8,16,32,64,128,256,512])
    cfg.define_knob("tile_x", [2,4,8,16,32,64,128,256,512])

    # 4. schedule according to config
    yo, yi = s[C].split(y, cfg["tile_y"].val)
    xo, xi = s[C].split(x, cfg["tile_x"].val)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


def main():
    N, L, M = 2028 , 2048 , 2048
    task = autotvm.task.create("test/tvmmatmul_v1", args=(N, L, M,"float64"), target="llvm")
    print(task.config_space)
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

 
        # Create a measurement callback with the custom function
    
    # measure_callbacks = [record_execution_time]

    measure_option = autotvm.measure_option(builder="local", 
                                            runner=autotvm.LocalRunner(number=1, repeat=1, timeout=200), # timeout=200
                                            )


    tuner = autotvm.tuner.GATuner(task)
    tuner.tune(
    n_trial=10,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("tvm_GATuner.log")]
    )

    with autotvm.apply_history_best("tvm_GATuner.log"):
        with tvm.target.Target("llvm"):
            s, arg_bufs = matmul_v1(N, L, M,"float32")
            func = tvm.build(s, arg_bufs)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()

    print("Elpased time = {}".format(end-start))