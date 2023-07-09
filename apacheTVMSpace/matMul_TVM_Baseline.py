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
    # print(C)
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    
    #using 8 as the tiling factor
    yo, yi = s[C].split(y, 8) 
    xo, xi = s[C].split(x, 8)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]




def main():
    N,L,M = 2048, 2048, 2048
    s, arg_bufs = matmul_basic(N, L, M, "float32")
    func = tvm.build(s,arg_bufs)
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = a_np.dot(b_np)
    c_tvm = tvm.nd.empty(c_np.shape)
  
    func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)



if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()

    print("Elpased time = {} secs".format(round(end-start,2)))