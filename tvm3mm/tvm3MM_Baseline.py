import numpy as np
import tvm
from tvm import autotvm

import time
from tvm import te



def matmul_basic(N, L, M, dtype):

    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    
    C = te.placeholder((N, L), name="C", dtype=dtype)
    D = te.placeholder((L, M), name="D", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    l = te.reduce_axis((0, L), name="l")
    m = te.reduce_axis((0, L), name="m")

    E = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="E")
    F = te.compute((N, M), lambda i, j: te.sum(C[i, l] * D[l, j], axis=l), name="F")
    G = te.compute((N, M), lambda i, j: te.sum(E[i, m] * F[m, j], axis=m), name="G")
    
    s1 = te.create_schedule(E.op)
    s2 = te.create_schedule(F.op)
    s3 = te.create_schedule(G.op)

    # schedule
    y, x = s1[E].op.axis
    k = s1[E].op.reduce_axis[0]
    y1, x1 = s2[F].op.axis
    l = s2[F].op.reduce_axis[0]
    y2, x2 = s3[G].op.axis
    m = s3[G].op.reduce_axis[0]
    
    #using 8 as the tiling factor
    yo, yi = s1[E].split(y, 8) 
    xo, xi = s1[E].split(x, 8)
    yo1, yi1 = s2[F].split(y1, 8) 
    xo1, xi1 = s2[F].split(x1, 8)
    yo2, yi2 = s3[G].split(y2, 8) 
    xo2, xi2 = s3[G].split(x2, 8)

    s1[E].reorder(yo, xo, k, yi, xi)
    s2[F].reorder(yo1, xo1, l, yi1, xi1)
    s3[G].reorder(yo2, xo2, m, yi2, xi2)

    return s3, [A, B, C, D, G]



def main():
    N,L,M = 2048, 2048, 2048
    s, arg_bufs = matmul_basic(N, L, M, "float64")
    func = tvm.build(s,arg_bufs)
    a_np = np.random.uniform(size=(N, L)).astype(np.float64)
    b_np = np.random.uniform(size=(L, M)).astype(np.float64)
    c_np = np.random.uniform(size=(N, L)).astype(np.float64)
    d_np = np.random.uniform(size=(L, M)).astype(np.float64)

    mm_np = a_np.dot(b_np).dot(c_np.dot(d_np))
    mm_tvm = tvm.nd.empty(mm_np.shape, dtype='float64')

    start = time.time()
    func(tvm.nd.array(a_np), tvm.nd.array(b_np),tvm.nd.array(c_np),tvm.nd.array(d_np), mm_tvm)
    end = time.time()
    print(mm_tvm)

    print("Elpased time = {} secs".format(round(end-start,2)))


if __name__ == '__main__':
    
    main()
    