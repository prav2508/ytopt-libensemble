import numpy as np
import tvm
from tvm import autotvm

import time
from tvm import te



def matmul_basic(N, L, M, dtype):

    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    k = te.reduce_axis((0, L), name="k")
    E = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="E")
    s = te.create_schedule(E.op)
    # schedule
    y, x = s[E].op.axis
    k = s[E].op.reduce_axis[0]
    # using 8 as the tiling factor
    yo, yi = s[E].split(y, 8) 
    xo, xi = s[E].split(x, 8)
    s[E].reorder(yo, xo, k, yi, xi)
    return s, [A, B, E]

@autotvm.template("trial/tvm3mm_v1") 
def runner(a,b,c,d,e,f,g):
    pass


def main():
    N,L,M = 2048, 2048, 2048
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = np.random.uniform(size=(N, L)).astype(np.float32)
    d_np = np.random.uniform(size=(L, M)).astype(np.float32)

    e_np = a_np.dot(b_np)
    f_np = c_np.dot(d_np)

    e_tvm = tvm.nd.empty(e_np.shape)
    f_tvm = tvm.nd.empty(f_np.shape)

    g_np = e_np.dot(f_np)
    g_tvm = tvm.nd.empty(g_np.shape)

    s, arg_bufs = matmul_basic(N, L, M, "float32")

    func = tvm.build(s,arg_bufs)

    start = time.time()
    func(tvm.nd.array(a_np), tvm.nd.array(b_np), e_tvm)
    func(tvm.nd.array(c_np), tvm.nd.array(d_np), f_tvm)
    func(e_tvm, f_tvm, g_tvm)
    end = time.time()
    print(g_tvm)
    print("Elpased time = {} secs".format(round(end-start,2)))



if __name__ == '__main__':
    
    main()
    