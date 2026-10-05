from numba import cuda, float64
import numpy as np
from time import perf_counter
from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "0"

# PANTELIS
#
# Logic for tiling:
# In the previous kernels, each thread read from global memory innerDim times
# per output element, so the same A_T / B element is fetched separately by many
# threads -> wasted bandwidth.
#
# Tiling: split the matrices into 32x32 tiles. One 32x32 block of threads:
#   1. Cooperatively loads one tile of A_T and one of B into shared memory
#      (each thread loads 1 element from each -> 1024*2 = 2048 global reads
#      per block, instead of 1024*32*2 = 65536 without tiling).
#   2. Threads compute partial dot products from the shared tiles (~100x
#      faster than global memory).
#   3. Advance to the next tile along the K dimension and repeat until K is
#      fully covered.
#   4. At the end, each thread writes its accumulated tempSum to C[row, col].
#
# The tricky part is which thread loads which element so that accesses are
# coalesced. The assignment hints: "map threadX to the column of A_T / B for
# perfect memory coalescing."
#
# Indexing:
# Grid mapping is x -> row, y -> col, so:
#   row = blockIdx.x * 32 + threadX
#   col = blockIdx.y * 32 + threadY
# Threads of the same warp have consecutive threadX and the same threadY
# (with this block shape, x is the fastest-varying dim in warp packing). So for
# coalescing, threads with consecutive threadX must hit consecutive global
# addresses.
#
# A_T tile (shape K x M): on iteration i we load
#   A_T[i*32:(i+1)*32, blockIdx_x*32:(blockIdx_x+1)*32]
#   - tile row    = i*32 + threadY    (position along K)
#   - tile column = blockIdx_x*32 + threadX  (position along M, == global row)
#   sharedMemA[threadY, threadX] = A_T[i*32 + threadY, row]
# Consecutive threadX -> A_T[..., row], A_T[..., row+1], ...  -> coalesced.
#
# B tile (shape K x N): on iteration i we load
#   B[i*32:(i+1)*32, blockIdx_y*32:(blockIdx_y+1)*32]
# Here col depends on threadY, not threadX, so we load with "swapped" indices
# to keep threadX aligned with the column of B:
#   sharedMemB[threadY, threadX] = B[i*32 + threadY, blockIdx_y*32 + threadX]
# Consecutive threadX -> B[..., col_base], B[..., col_base+1], ... -> coalesced.
#
# Compute: for C[row, col], the per-tile dot product is
#   sum over k_local of A_T[k, row] * B[k, col]
# With the shared layout above:
#   A_T[k, row] -> sharedMemA[k_local, threadX]   (row tied to threadX)
#   B[k, col]   -> sharedMemB[k_local, threadY]   (col tied to threadY)
# So the inner loop is:
#   tempSum += sharedMemA[k, threadX] * sharedMemB[k, threadY]   for k in range(32)


# Use the naive grid mapping where x corresponds to rows, and y to columns 
threadGridDim = 32
threadsPerBlock = (threadGridDim, threadGridDim)

@cuda.jit("void(float64[:,:], float64[:,:],  float64[:,:])")
def SharedMemMatMul2DKernel(A_T, B, C):
    # 2D thread grid
    # get thread coords from cuda.grid (returns absolute position of the current thread in the entire grid of blocks)
    row, col = cuda.grid(2)
    
    threadX = cuda.threadIdx.x
    threadY = cuda.threadIdx.y

    blockIdx_x = cuda.blockIdx.x
    blockIdx_y = cuda.blockIdx.y

    # Create the shared memory arrays
    sharedMemA = cuda.shared.array((threadGridDim, threadGridDim), dtype=float64)
    sharedMemB = cuda.shared.array((threadGridDim, threadGridDim), dtype=float64)
    
    # Number of chunks in the K dimension
    k_chunks = int(B.shape[0] / threadGridDim)
    if B.shape[0] % threadGridDim != 0:
        k_chunks += 1

    K = B.shape[0]
    M = C.shape[0]   # numRows
    N = C.shape[1]   # numCols

    tempSum = 0.0
    
    for i in range(k_chunks):
        # 1. COALESCED LOAD OF A_T chunk
        # A_T is (K, M). We load a 32x32 tile. 
        # By mapping threadX to the column of A_T, we get perfect memory coalescing!
        
        # A_T has shape (K, M). Tile column = global row = blockIdx_x*32 + threadX.
        # Mapping threadX -> column of A_T gives coalesced reads across the warp
        a_k = i * threadGridDim + threadY
        a_m = blockIdx_x * threadGridDim + threadX
        if a_k < K and a_m < M:
            sharedMemA[threadY, threadX] = A_T[a_k, a_m]
        else:
            sharedMemA[threadY, threadX] = 0.0
            
        # 2. COALESCED LOAD OF B chunk
        # B is (K, N). We load a 32x32 tile.
        # By mapping threadX to the column of B, we get perfect memory coalescing!
        
        # B has shape (K, N). Tile column = blockIdx_y*32 + threadX.
        # Same: threadX -> column of B for coalescing.
        b_k = i * threadGridDim + threadY
        b_n = blockIdx_y * threadGridDim + threadX
        if b_k < K and b_n < N:
            sharedMemB[threadY, threadX] = B[b_k, b_n]
        else:
            sharedMemB[threadY, threadX] = 0.0
            
        # Do not forget to synchronize threads of each block after loading A_T and B chunks!
        cuda.syncthreads()
        
        # 3. COMPUTE TILE DOT PRODUCT
        # This thread computes the output for global 'row' and 'col'.
        # 'row' corresponds to threadX in the A tile.
        # 'col' corresponds to threadY in the B tile.
        for k in range(threadGridDim):
            tempSum += sharedMemA[k, threadX] * sharedMemB[k, threadY]

            
        # Do not forget to synchronize threads of each block after computing the tile dot product!
        cuda.syncthreads()
        
    # 4. WRITE TO C (Write result, with boundary check)
    if row < M and col < N:
        C[row, col] = tempSum


def CudaSharedMemMatMul(A, B):

    numRows = A.shape[0]
    numCols = B.shape[1]

    # 5. Transfer A and B to device memory (A_T_device, B_device) and initialize C in device memory (C_device)
    # Physically transpose A on the CPU so it becomes contiguous in memory
    tic = perf_counter()

    # Transpose A on the CPU so the kernel can read it with coalesced patterns,
    # then push everything to the device.
    A_T = np.ascontiguousarray(A.T)
    A_T_device = cuda.to_device(A_T)
    B_device = cuda.to_device(B)
    C_device = cuda.device_array((numRows, numCols), dtype=np.float64)

    toc = perf_counter()
    transfersTime = toc-tic

    # 6. Use the naive grid mapping where x corresponds to rows, and y to columns #
    blocksPerGrid_x = (numRows + threadGridDim - 1) // threadGridDim
    blocksPerGrid_y = (numCols + threadGridDim - 1) // threadGridDim
    
    blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)

    # 7. Call the kernel
    # Do not forget to synchronize!
    SharedMemMatMul2DKernel[blocksPerGrid, threadsPerBlock](A_T_device, B_device, C_device)
    cuda.synchronize()
    
    # 8. Copy C back to host
    tic = perf_counter()
    C = C_device.copy_to_host()
    toc = perf_counter()
    transfersTime += toc - tic

    return C, transfersTime



# Notes:
# - Both syncthreads() are required. The first ensures all threads finished
#   loading before any reads from shared memory; the second ensures all threads
#   finished computing before the next iteration overwrites the shared tile.
#   Skipping the second causes wrong results only on large matrices (warp
#   divergence) -> nasty to debug.
# - Boundary handling via zero-padding. When K (or M, N) isn't a multiple of 32,
#   out-of-range loads write 0.0 into shared memory so the dot product stays
#   correct (x * 0 = 0). Avoids branches in the compute loop.
# - Coalescing on both tiles. Common bug: writing sharedMemA[threadX, threadY]
#   instead of [threadY, threadX], or flipping indices on only one tile. Rule:
#   threads with consecutive threadX must always hit consecutive global addresses.
# - No bank conflicts in the compute loop. sharedMemA[k, threadX]: consecutive
#   threadX -> different banks (no conflict). sharedMemB[k, threadY]: consecutive
#   threadX share threadY -> same bank, broadcast (also no conflict).
# - When the benefit shows. vs transpose-only, expect 2-4x on large matrices
#   (memory-bound). On small matrices or small innerDim (e.g. MNIST's 784 input
#   layer, ~24 outer iterations) the gain is smaller but still the fastest of
#   the three.
# - Static threadGridDim. Numba requires compile-time-known shared array size,
#   so the block dim is hardcoded as the global threadGridDim = 32. Changing
#   the block size means changing this constant too.