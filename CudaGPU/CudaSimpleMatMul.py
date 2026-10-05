from numba import cuda
import numpy as np
from time import perf_counter
from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "0"

@cuda.jit("void(float64[:,:], float64[:,:],  float64[:,:])")
def SimpleMatMul2DKernel(A, B, C):
    # 2D thread grid
    # get thread coords from cuda.grid (returns absolute position of the current thread in the entire grid of blocks)
    row, col = cuda.grid(2)

    numRows = C.shape[0]
    numCols = C.shape[1]
    innerDim = A.shape[1]
    
    # TASK: The code that thread (row,col) will execute (calculating C[row][col])
    # Boundary check: some threads on the edge of the grid fall outside C
    if row < numRows and col < numCols:
        tempSum = 0.0
        for k in range(innerDim):
            tempSum += A[row, k] * B[k, col]
        C[row, col] = tempSum

def CudaSimpleMatMul(A, B):

    numRows = A.shape[0]
    numCols = B.shape[1]
    
    # 1. TASK: Transfer A and B to device memory (A_device, B_device) and initialize C in device memory (C_device)
    tic = perf_counter()
    
    A_device =  cuda.to_device(A)
    B_device = cuda.to_device(B)
    C_device = cuda.device_array((numRows, numCols), dtype=np.float64)

    toc = perf_counter()
    transfersTime = toc-tic

    # 2. TASK: Create a grid of thread blocks and fire the kernel. You should use a 2D grid (of 2D blocks)
    threadGridDim = 32 #chooses the block size (number of threads per block in each dimension). 32 is a common choice for CUDA kernels.
    threadsPerBlock = (threadGridDim, threadGridDim) #each block is a 2D

    
    # Round up so the grid fully covers C, even when dims aren't multiples of 32
    blocksPerGrid_x = (numRows + threadGridDim - 1) // threadGridDim
    blocksPerGrid_y = (numCols + threadGridDim - 1) // threadGridDim

    blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
    
    # 3. TASK: Call the kernel
    SimpleMatMul2DKernel[blocksPerGrid, threadsPerBlock](A_device, B_device, C_device)
    
    # 4. TASK: Do not forget to synchronize!
    cuda.synchronize()

    # 5. TASK: Copy result back to host
    tic = perf_counter()
    C = C_device.copy_to_host()
    toc = perf_counter()
    transfersTime += toc - tic

    return C, transfersTime
