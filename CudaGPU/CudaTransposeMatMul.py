from numba import cuda
import numpy as np
from time import perf_counter
from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "0"

# PANTELIS
# Memory Access Pattern Optimization via transpose of A
#
# Why transpose: In CudaSimpleMatMul, threads of the same warp typically have consecutive `col` and the same `row`. In the inner loop:
#   - B[k, col]: neighbouring threads read B[k, col], B[k, col+1], ... -> coalesced.
#   - A[row, k]: all threads read the same address -> broadcast, fine.
#
# Here, the grid mapping is flipped (x -> row, y -> col), so threads of the same warp have consecutive `row` and the same `col`. Now:
#   - B[k, col]: all threads read the same address -> broadcast, fine.
#   - A[row, k]: neighbouring threads read A[row, k], A[row+1, k], ... which are
#     `innerDim` elements apart in row-major memory -> uncoalesced, very slow.
#
# Fix: physically transpose A on the CPU so A_T[k, row] == A[row, k]. The kernel
# then reads A_T[k, row], and neighbouring threads (consecutive `row`) hit
# consecutive addresses -> coalesced again.


@cuda.jit("void(float64[:,:], float64[:,:],  float64[:,:])")
def TransposeMatMul2DKernel(A_T, B, C):
    # 1. 2D thread grid: x -> row, y -> col
    row, col = cuda.grid(2)   

    # Get thread coords from cuda.grid (returns absolute position of the current thread in the entire grid of blocks)
    numRows = C.shape[0]
    numCols = C.shape[1]
    innerDim = B.shape[0]       # also equals A_T.shape[0]
    
    # 2. Check boundaries against C (numRows x numCols)
    #    The code that thread (row,col) will execute (calculating C[row][col])
    if row < numRows and col < numCols:
        tempSum = 0.0
        # A_T[k, row] == A[row, k], so neighbouring threads (consecutive row)
        # access consecutive memory in A_T -> coalesced reads
        for k in range(innerDim):
            tempSum += A_T[k, row] * B[k, col]
        C[row, col] = tempSum

def CudaTransposeMatMul(A, B):

    numRows = A.shape[0]
    numCols = B.shape[1]
    
    # 3. Transfer A and B to device memory (A_T_device, B_device) and initialize C in device memory (C_device)
    tic = perf_counter()

    # Physically transpose A on the CPU so the transposed array is contiguous in memory: 
    # np.ascontiguousarray forces an actual copy with the new layout (a plain .T would just return a non-contiguous view).
    A_T = np.ascontiguousarray(A.T)
    A_T_device = cuda.to_device(A_T)
    B_device = cuda.to_device(B)
    C_device = cuda.device_array((numRows, numCols), dtype=np.float64)

    toc = perf_counter()
    transfersTime = toc-tic

    # 4.: Use the naive grid mapping where x corresponds to rows, and y to columns #
    threadGridDim = 32
    threadsPerBlock = (threadGridDim, threadGridDim)
    
    # 5. Create a grid of thread blocks and fire the kernel. You should use a 2D grid (of 2D blocks) #
    blocksPerGrid_x = (numRows + threadGridDim - 1) // threadGridDim
    blocksPerGrid_y = (numCols + threadGridDim - 1) // threadGridDim
    
    blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
    
    # 6. Call the kernel
    # Do not forget to synchronize!
    TransposeMatMul2DKernel[blocksPerGrid, threadsPerBlock](A_T_device, B_device, C_device)
    cuda.synchronize()

    # 7. Copy C back to host
    tic = perf_counter()
    C = C_device.copy_to_host()
    toc = perf_counter()
    transfersTime += toc - tic

    return C, transfersTime


# Notes:
# - np.ascontiguousarray(A.T) instead of plain A.T: A.T is just a view with swapped strides, same physical layout. Sending it to the GPU gives the
#   same uncoalesced accesses. ascontiguousarray forces a real row-major copy.
# - Transpose is included in transfersTime: the CPU transpose costs O(numRows * innerDim) and must be charged to the GPU solution for a fair
#   comparison (the assignment puts it under the same timer).
# - Grid mapping x -> row, y -> col is the same as CudaSimpleMatMul. Only the data layout changes here, not the thread layout. So blocksPerGrid_x comes
#   from numRows and _y from numCols, exactly as before.
# - When the benefit shows up: for small matrices or ones that fit in L2, the simple vs transpose gap is tiny (L2 hides uncoalesced accesses). For large
#   matrices (e.g. MNIST's 70000 x 784 x 128) the speedup is clear.
# - Correctness unchanged: A_T[k, row] == A[row, k], so the math is identical. np.allclose(C_simple, C_transpose) must pass.