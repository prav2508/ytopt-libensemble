import numpy as np

def matrixMultiply(N, L, M):


    A = np.random.rand(N, L)
    B = np.random.rand(L, M)
    
    tile_x = 100
    tile_y = 100

    # Perform matrix multiplication with tiling
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(0, A.shape[0], tile_x):
        for j in range(0, B.shape[1], tile_y):
            for k in range(0, A.shape[1], tile_y):
                C[i:i+tile_x, j:j+tile_y] += np.dot(A[i:i+tile_x, k:k+tile_y], B[k:k+tile_y, j:j+tile_y])
                
    return C


def main():
    print(matrixMultiply(2048,2048,2048))



if __name__ == '__main__':
    main()