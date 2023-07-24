import numpy as np
import tvm
from tvm import autotvm

import time
from tvm import te

def matmul_basic(N, L, M, dtype):

    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    s = te.create_schedule(C.op)


    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    

    yo, yi = s[C].split(y, #P0) 
    xo, xi = s[C].split(x, #P1)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]




def main():
    # N, L, M = 1000 , 1100 , 1200
    N, L, M = 2000 , 2300 , 2600
    s, arg_bufs = matmul_basic(N, L, M, "float64")
    func = tvm.build(s,arg_bufs)
    a_np = np.random.uniform(size=(N, L)).astype(np.float64)
    b_np = np.random.uniform(size=(L, M)).astype(np.float64)
    c_np = a_np.dot(b_np)
    
    c_tvm = tvm.nd.empty(c_np.shape, dtype="float64")

    start = time.time()
    func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)
    end = time.time()
    print("Elapsed time = {}".format(round(end-start,2)))

if __name__ == '__main__':
    
    main()
    

    