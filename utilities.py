import numpy as np
from Sequential.SequentialMatMul import SequentialMatMul
from Multiprocessing.MultiprocessingMatMul import MultiprocessingMatMul
from Multiprocessing.tools import ReleaseSharedMemory
from CudaGPU.CudaSharedMemMatMul import CudaSharedMemMatMul
from CudaGPU.CudaSimpleMatMul import CudaSimpleMatMul
from CudaGPU.CudaTransposeMatMul import CudaTransposeMatMul
from time import perf_counter

def CustomMatmul(A, B, mode='numpy', numberOfProcesses=1):
    
    if mode == 'sequential':
        out, execTime = SequentialMatMul(A, B)
        return out, execTime
    
    if mode == 'numpy' or len(A.shape) == 3:
        # print(A.shape, 'x', B.shape)
        tic = perf_counter()
        out = np.matmul(A, B)
        toc = perf_counter()
        numpyTestTime = toc - tic
        return out, numpyTestTime

    if mode == 'multiprocessing':
        out, execTime = MultiprocessingMatMul(A, B, numberOfProcesses=numberOfProcesses)
        return out, execTime

    if mode == 'cudaSimple':
        out = CudaSimpleMatMul(A, B)
        return out

    if mode == 'cudaTranspose':
        out = CudaTransposeMatMul(A, B)
        return out    

    if mode == 'cudaShmem':
        out = CudaSharedMemMatMul(A, B)
        # if not np.allclose(out[0], np.matmul(A,B)):
        #     print(40*'-')
        #     print(mode)
        #     print('out', out[0])
        #     print(np.matmul(A,B))
        #     print(40*'-')
        return out
    else:
        raise Exception(f"Unknown MatMul mode: {mode}")


def load_data(path,datatype='float64'):
    with np.load(path) as f:
        x_train, y_train = f['x_train'].astype(datatype), f['y_train'].astype(datatype)
        x_test, y_test = f['x_test'].astype(datatype), f['y_test'].astype(datatype)
        return (x_train, y_train), (x_test, y_test)
