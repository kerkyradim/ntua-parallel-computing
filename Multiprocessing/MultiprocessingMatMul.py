from copy import deepcopy
from multiprocessing import Process, shared_memory, cpu_count
from Multiprocessing.tools import ReleaseSharedMemory, CreateSharedNumpyArray
import numpy as np
from time import perf_counter

# Logic:
# Distribute the rows. If numRows = 100 and numberOfProcesses = 4, then
# numRowsPerProcess = 100 // 4 = 25. Process i handles rows [i*25, (i+1)*25).
# When numRows isn't divisible evenly (e.g. 103 rows, 4 processes), the first
# 3 processes take 25 rows each and the last takes the remainder (28). The
# pre-written `if i == numberOfProcesses - 1: iEndIndex = A.shape[0]` handles this.
#
# Target function: each process runs the same triple loop as the sequential
# version, but only for row in range(iStartIndex, iEndIndex), writing into shared C.
#
# Why not numpy.dot: the assignment forbids it. On single row x single col,
# np.dot's overhead exceeds a plain Python loop, and offloading the inner
# product to BLAS would hide the actual parallel work from the processes.

def MultiprocessingTargetFunction(A, B, C, iStartIndex, iEndIndex):
    numCols = B.shape[1]
    innerDim = B.shape[0]

	# Perform GEMM for the rows assigned to each process
    for i in range(iStartIndex, iEndIndex):
        for j in range(numCols):
            # Compute dot product of A[i, :] with B[:, j]
            for k in range(innerDim):
                C[i, j] += A[i, k] * B[k, j]

    return

def MultiprocessingMatMul(A, B, numberOfProcesses=1):
    outArrayShape = (A.shape[0], B.shape[1])

    CreateSharedNumpyArray(outArrayShape, arrayName='C')
    shm = shared_memory.SharedMemory(name='C')
    C = np.ndarray(outArrayShape, dtype=np.float64, buffer=shm.buf)
    
    # Get Number of Rows to be computed by each process (integer div)
    # 1. TASK: # the remainder goes to the last process via the bounds check below)
    numRowsPerProcess = A.shape[0] // numberOfProcesses

    tic = perf_counter()

    processes = []
    for i in range(numberOfProcesses):

        # 2. TASK: Get Start and End Index for each process
        # Each process gets a contiguous chunk of rows
        iStartIndex= i * numRowsPerProcess
        iEndIndex = (i + 1) * numRowsPerProcess

        # Do not get out of bounds for the last process
        if i == numberOfProcesses - 1:
            iEndIndex = A.shape[0]

        iProcess = Process(target=MultiprocessingTargetFunction, args=(A, B, C, iStartIndex, iEndIndex,))

        processes.append(iProcess)
        iProcess.start()

    for iProcess in processes:
        iProcess.join()

    toc = perf_counter()
    execTime = toc - tic

    outC = deepcopy(C)
    
    ReleaseSharedMemory(['C'])
    
    return outC, execTime


# Notes:
# - Shared memory only for C: A and B are pickled to each process via Process(args=...),
#   which adds overhead for large matrices. Visible in benchmarks but not a bug.
# - No race condition: each process writes to disjoint rows of C, so no locks needed.
# - Load balancing: when numRows isn't divisible by numberOfProcesses, the last
#   process gets the remainder and can become a straggler. Fine for MNIST dims.
# - Process startup overhead: fork/spawn costs tens of ms per process. For small
#   matrices, startup can dominate the work => poor or negative speedup (expected).
# - deepcopy(C) before release: C is a view over shared memory that gets freed
#   right after; returning it directly would hand the caller invalid memory.